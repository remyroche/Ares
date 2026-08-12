"""Strict-R3 common-bps EV bridge for model-vintage transitions.

The monthly strict-R3 upstream model is intentionally re-fitted.  Its raw
scores must therefore never be pooled with an older producer for a live EV
map.  At the same time, an exact-producer-only map has a cold start whenever a
new producer appears: it may mechanically reject every candidate for several
days despite complete, valid decision-time inputs.

This module resolves that tension in two explicitly separated stages:

``same-producer prior-42 CDF score``
    -> ``frozen, strict-OOF common-bps prior``
    -> ``causal 21/42/84-day side-local residual correction``

Only the *residual in policy-net bps* crosses a model-vintage boundary.  The
recent correction never sees or ranks raw scores, which preserves the rule
that raw score domains are not bridgeable.  A newly fitted producer therefore
starts from the frozen prior with a zero recent correction rather than from an
unmapped/fail-closed state.

The bridge is an inference artifact.  It must be fit on strict-OOF rows whose
policy outcomes resolved before its declared cutoff.  It is not fitted on the
live or evaluation period that it scores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .strict_r3_canonical_v2 import BinnedPolicyNetMap, _file_hash, fit_policy_net_map


BRIDGE_SCHEMA = "strict_r3_ev_bridge_v1_common_bps_residual"
COMMON_BPS_CALIBRATION_MODE = "strict_oof_common_bps_bridge_plus_causal_residual_v1"
EXACT_PRODUCER_RESERVE_CALIBRATION_MODE = (
    "strict_oof_exact_producer_reserve_map_plus_causal_residual_v1"
)
DEFAULT_SCORE_COLUMN = "final_score"
DEFAULT_NET_COLUMN = "policy_net_bps"
DEFAULT_DECISION_COLUMN = "__decision_ts__"
DEFAULT_LABEL_AVAILABLE_COLUMN = "policy_label_available_ts"
DEFAULT_IDENTITY_COLUMN = "candidate_id"
SIDES = ("long", "short")


@dataclass(frozen=True)
class EVBridgeSpec:
    """Frozen common-bps prior plus conservative recent-residual settings."""

    prior_bins: int = 20
    prior_trim_fraction: float = 0.05
    required_prior_rows_per_side: int = 100
    residual_windows_days: tuple[int, ...] = (84, 42, 21)
    residual_shrinkage_rows: tuple[float, ...] = (200.0, 100.0, 50.0)
    residual_trim_fraction: float = 0.05
    minimum_residual_rows: int = 20
    net_floor_bps: float = 50.0

    def __post_init__(self) -> None:
        if self.prior_bins < 4:
            raise ValueError("EV bridge needs at least four prior bins")
        if self.required_prior_rows_per_side < max(100, self.prior_bins * 4):
            raise ValueError("EV bridge prior support is too small")
        if tuple(sorted(self.residual_windows_days, reverse=True)) != self.residual_windows_days:
            raise ValueError("residual windows must be in decreasing broad-to-recent order")
        if len(self.residual_windows_days) != len(self.residual_shrinkage_rows):
            raise ValueError("one residual shrinkage value is required per window")
        if any(window <= 0 for window in self.residual_windows_days):
            raise ValueError("residual windows must be positive")
        if any(value <= 0.0 for value in self.residual_shrinkage_rows):
            raise ValueError("residual shrinkage values must be positive")
        if self.minimum_residual_rows < 4:
            raise ValueError("residual correction needs at least four rows")
        if not 0.0 <= self.prior_trim_fraction < 0.5:
            raise ValueError("prior trim fraction must be in [0, .5)")
        if not 0.0 <= self.residual_trim_fraction < 0.5:
            raise ValueError("residual trim fraction must be in [0, .5)")


@dataclass
class StrictR3EVBridgeBundle:
    """Immutable, OOF common-bps calibration artifact."""

    fit_cutoff: pd.Timestamp
    score_column: str
    net_column: str
    ev_score_family_id: str
    geometry_bundle_sha256: str
    side_maps: dict[str, BinnedPolicyNetMap]
    producer_lineage: Mapping[str, str] = field(default_factory=dict)
    # Exact-producer cell-day calibration state.  ``side_score_references``
    # retains the complete sorted reserve score domain so live candidates are
    # assigned to exactly the same fixed twenty cells as the historical
    # ablation.  ``cell_day_seed`` is the compact one-row-per-side/day/cell
    # policy-net mean from that reserve; it contains no candidate features and
    # is never refit while the producer is active.
    side_score_references: dict[str, np.ndarray] = field(default_factory=dict)
    cell_day_seed: pd.DataFrame = field(default_factory=pd.DataFrame)
    spec: EVBridgeSpec = field(default_factory=EVBridgeSpec)
    manifest: dict[str, Any] = field(default_factory=dict)
    schema: str = BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        self.fit_cutoff = _utc(self.fit_cutoff)
        if self.schema != BRIDGE_SCHEMA:
            raise ValueError("wrong strict-R3 EV bridge schema")
        if not self.ev_score_family_id or not self.geometry_bundle_sha256:
            raise ValueError("EV bridge requires frozen score-family and geometry identities")
        if not set(self.side_maps).issubset(SIDES) or not self.side_maps:
            raise ValueError("EV bridge has no canonical side prior")
        if any(not isinstance(value, BinnedPolicyNetMap) for value in self.side_maps.values()):
            raise TypeError("EV bridge side maps have the wrong type")
        if not set(self.producer_lineage).issubset({
            "conversion_bundle_sha256", "upstream_bundle_sha256",
        }):
            raise ValueError("EV bridge producer lineage has unsupported keys")
        if any(not str(value) for value in self.producer_lineage.values()):
            raise ValueError("EV bridge producer lineage cannot contain empty identities")
        references = getattr(self, "side_score_references", {})
        if references:
            if not set(references).issubset(SIDES):
                raise ValueError("EV bridge has unsupported cell-day side references")
            for side, values in references.items():
                array = np.asarray(values, dtype=float)
                if len(array) < self.spec.prior_bins * 4 or not np.isfinite(array).all():
                    raise ValueError(f"EV bridge {side} cell reference lacks support")
                if np.any(array[1:] < array[:-1]):
                    raise ValueError(f"EV bridge {side} cell reference is not sorted")
        seed = getattr(self, "cell_day_seed", pd.DataFrame())
        if not seed.empty:
            required_seed = {"side_name", "__day__", "__cell__", "cell_day_ev_bps"}
            missing = sorted(required_seed.difference(seed.columns))
            if missing:
                raise ValueError(f"EV bridge cell-day seed lacks: {missing}")
            if not seed["side_name"].astype(str).str.lower().isin(SIDES).all():
                raise ValueError("EV bridge cell-day seed has an unsupported side")

    def predict_prior(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a frozen common-bps prior without using outcomes."""
        _validate_score_semantics(
            frame,
            ev_score_family_id=self.ev_score_family_id,
            geometry_bundle_sha256=self.geometry_bundle_sha256,
            score_column=self.score_column,
            require_prequential=True,
        )
        side = frame["side_name"].astype(str).str.lower().to_numpy(object)
        score = pd.to_numeric(frame[self.score_column], errors="coerce").to_numpy(float)
        output = np.full(len(frame), np.nan, dtype=float)
        for name, mapping in self.side_maps.items():
            positions = np.flatnonzero(side == name)
            if len(positions):
                output[positions] = mapping.predict(score[positions])
        return output

    @property
    def calibration_mode(self) -> str:
        """Distinguish an exact-producer reserve map from a fallback bridge.

        The serialized class and schema are deliberately retained for backward
        compatibility.  Producer lineage, however, makes the economic
        contract materially different: an exact reserve map never transfers a
        score-to-bps relationship between fitted producers.
        """
        return (
            EXACT_PRODUCER_RESERVE_CALIBRATION_MODE
            if self.producer_lineage
            else COMMON_BPS_CALIBRATION_MODE
        )


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(),
    ).hexdigest()


def _cell_day_state_identity(bundle: "StrictR3EVBridgeBundle") -> str | None:
    """Hash the inference-critical fixed-cell state independently of joblib."""
    references = getattr(bundle, "side_score_references", {})
    seed = getattr(bundle, "cell_day_seed", pd.DataFrame())
    if not references or seed.empty:
        return None
    reference_hashes = {
        side: hashlib.sha256(
            np.asarray(values, dtype=np.float64).tobytes(order="C"),
        ).hexdigest()
        for side, values in sorted(references.items())
    }
    ordered = seed.sort_values(
        ["side_name", "__day__", "__cell__"], kind="stable",
    ).reset_index(drop=True)
    seed_hash = hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).to_numpy(np.uint64).tobytes(),
    ).hexdigest()
    return _json_hash({"references": reference_hashes, "seed": seed_hash})


def _validate_score_semantics(
    frame: pd.DataFrame,
    *,
    ev_score_family_id: str,
    geometry_bundle_sha256: str,
    score_column: str,
    require_prequential: bool,
) -> None:
    required = {
        DEFAULT_IDENTITY_COLUMN, DEFAULT_DECISION_COLUMN, "side_name", score_column,
        "ev_score_family_id", "geometry_bundle_sha256", "stack_is_prequential",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EV bridge score frame lacks: {missing}")
    if frame[DEFAULT_IDENTITY_COLUMN].isna().any() or frame[DEFAULT_IDENTITY_COLUMN].duplicated().any():
        raise ValueError("EV bridge requires immutable unique candidate identities")
    if not frame["side_name"].astype(str).str.lower().isin(SIDES).all():
        raise ValueError("EV bridge requires canonical long/short sides")
    if not frame["ev_score_family_id"].astype(str).eq(str(ev_score_family_id)).all():
        raise ValueError("EV bridge rejects a different score-family semantic contract")
    if not frame["geometry_bundle_sha256"].astype(str).eq(str(geometry_bundle_sha256)).all():
        raise ValueError("EV bridge rejects a different geometry/K9 semantic contract")
    if require_prequential and not frame["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("EV bridge requires strict prequential score rows")


def fit_strict_r3_ev_bridge(
    ledger: pd.DataFrame,
    *,
    fit_cutoff: Any,
    score_column: str = DEFAULT_SCORE_COLUMN,
    net_column: str = DEFAULT_NET_COLUMN,
    decision_column: str = DEFAULT_DECISION_COLUMN,
    label_available_column: str = DEFAULT_LABEL_AVAILABLE_COLUMN,
    spec: EVBridgeSpec = EVBridgeSpec(),
    producer_lineage: Mapping[str, str] | None = None,
) -> StrictR3EVBridgeBundle:
    """Fit a frozen common-bps prior from only pre-cutoff strict-OOF rows."""
    required = {
        DEFAULT_IDENTITY_COLUMN, decision_column, "side_name", score_column, net_column,
        label_available_column, "ev_score_family_id", "geometry_bundle_sha256",
        "stack_is_prequential",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"EV bridge fit ledger lacks: {missing}")
    cutoff = _utc(fit_cutoff)
    work = ledger.copy()
    work[decision_column] = pd.to_datetime(work[decision_column], utc=True, errors="raise")
    work[label_available_column] = pd.to_datetime(work[label_available_column], utc=True, errors="raise")
    if work[DEFAULT_IDENTITY_COLUMN].isna().any() or work[DEFAULT_IDENTITY_COLUMN].duplicated().any():
        raise ValueError("EV bridge fit requires immutable unique candidate identities")
    family = work["ev_score_family_id"].dropna().astype(str).unique()
    geometry = work["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(family) != 1 or len(geometry) != 1:
        raise ValueError("EV bridge fit must use one score-family and frozen geometry contract")
    _validate_score_semantics(
        work,
        ev_score_family_id=str(family[0]),
        geometry_bundle_sha256=str(geometry[0]),
        score_column=score_column,
        require_prequential=True,
    )
    score = pd.to_numeric(work[score_column], errors="coerce")
    target = pd.to_numeric(work[net_column], errors="coerce")
    valid_path = (
        work["policy_path_valid"].fillna(False).astype(bool)
        if "policy_path_valid" in work else pd.Series(True, index=work.index)
    )
    fit = work.loc[
        work[label_available_column].lt(cutoff)
        & valid_path & np.isfinite(score) & np.isfinite(target)
    ].copy()
    if fit.empty:
        raise ValueError("EV bridge fit has no resolved OOF policy outcomes before cutoff")
    side_maps: dict[str, BinnedPolicyNetMap] = {}
    side_score_references: dict[str, np.ndarray] = {}
    cell_day_parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for side in SIDES:
        block = fit.loc[fit["side_name"].astype(str).str.lower().eq(side)]
        if len(block) < spec.required_prior_rows_per_side:
            continue
        mapping = fit_policy_net_map(
            block[score_column], block[net_column],
            bins=spec.prior_bins, trim_fraction=spec.prior_trim_fraction,
        )
        side_maps[side] = mapping
        reference_score = np.sort(
            pd.to_numeric(block[score_column], errors="coerce").to_numpy(float),
            kind="stable",
        )
        side_score_references[side] = reference_score
        block_score = pd.to_numeric(
            block[score_column], errors="coerce",
        ).to_numpy(float)
        cell = np.minimum(
            np.searchsorted(reference_score, block_score, side="right")
            * spec.prior_bins // len(reference_score),
            spec.prior_bins - 1,
        ).astype(np.int16)
        seed = pd.DataFrame({
            "side_name": side,
            "__day__": pd.to_datetime(
                block[decision_column], utc=True, errors="raise",
            ).dt.normalize().to_numpy(),
            "__cell__": cell,
            "policy_net_bps": pd.to_numeric(
                block[net_column], errors="coerce",
            ).to_numpy(float),
        })
        cell_day_parts.append(
            seed.groupby(
                ["side_name", "__day__", "__cell__"],
                observed=True, sort=True,
            ).agg(
                cell_day_ev_bps=("policy_net_bps", "mean"),
                cell_day_trades=("policy_net_bps", "size"),
            ).reset_index()
        )
        audit.append({
            "side_name": side,
            "fit_rows": int(len(block)),
            "fit_min_label_available_ts": block[label_available_column].min().isoformat(),
            "fit_max_label_available_ts": block[label_available_column].max().isoformat(),
            "score_bin_count": int(len(mapping.bin_x)),
            "score_bin_support": mapping.bin_support.astype(int).tolist(),
            "prior_curve_bps": mapping.bin_y.astype(float).tolist(),
        })
    if not side_maps:
        raise ValueError("EV bridge fit lacks side support")
    lineage = {str(key): str(value) for key, value in (producer_lineage or {}).items()}
    exact_producer_reserve = bool(lineage)
    manifest = {
        "schema": BRIDGE_SCHEMA,
        "fit_cutoff": cutoff.isoformat(),
        "fit_rule": (
            "strict-OOF score, valid policy path, and policy label available "
            "before cutoff"
        ),
        "score_coordinate": "same-producer prior-42 CDF, not raw upstream score",
        "score_column": score_column,
        "net_column": net_column,
        "ev_score_family_id": str(family[0]),
        "geometry_bundle_sha256": str(geometry[0]),
        "side_fit_audit": audit,
        "cell_day_calibration": {
            "score_cells": spec.prior_bins,
            "score_reference": "complete sorted exact-producer OOS reserve score domain",
            "seed_weighting": "one mean policy-net observation per UTC day x fixed score cell",
            "seed_rows": int(sum(len(value) for value in cell_day_parts)),
        },
        "calibration_mode": (
            EXACT_PRODUCER_RESERVE_CALIBRATION_MODE
            if exact_producer_reserve else COMMON_BPS_CALIBRATION_MODE
        ),
        "calibration_role": (
            "same exact producer OOS reserve map; no cross-vintage mapping"
            if exact_producer_reserve
            else "common-bps fallback prior; no raw score pooling across producer vintages"
        ),
        "residual_rule": (
            "same-producer realised-policy residual"
            if exact_producer_reserve
            else "common-bps realised-policy residual; no raw score pooling across producer vintages"
        ),
        "producer_lineage": lineage,
        "spec": {
            "prior_bins": spec.prior_bins,
            "prior_trim_fraction": spec.prior_trim_fraction,
            "residual_windows_days": list(spec.residual_windows_days),
            "residual_shrinkage_rows": list(spec.residual_shrinkage_rows),
            "minimum_residual_rows": spec.minimum_residual_rows,
            "net_floor_bps": spec.net_floor_bps,
        },
    }
    return StrictR3EVBridgeBundle(
        fit_cutoff=cutoff,
        score_column=score_column,
        net_column=net_column,
        ev_score_family_id=str(family[0]),
        geometry_bundle_sha256=str(geometry[0]),
        side_maps=side_maps,
        producer_lineage=lineage,
        side_score_references=side_score_references,
        cell_day_seed=(
            pd.concat(cell_day_parts, ignore_index=True)
            if cell_day_parts else pd.DataFrame()
        ),
        spec=spec,
        manifest=manifest,
    )


def persist_strict_r3_ev_bridge(
    bundle: StrictR3EVBridgeBundle, directory: Path,
) -> Mapping[str, object]:
    """Persist an immutable bridge artifact with an auditable identity."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable EV bridge directory already exists: {directory}")
    directory.mkdir(parents=True)
    payload = directory / "strict_r3_ev_bridge.joblib"
    joblib.dump(bundle, payload, compress=3)
    digest = _file_hash(payload)
    manifest = {
        **bundle.manifest,
        "bundle_file": payload.name,
        "bundle_sha256": digest,
        "cell_day_state_identity": _cell_day_state_identity(bundle),
        "bundle_identity": _json_hash({
            "schema": bundle.schema,
            "fit_cutoff": bundle.fit_cutoff.isoformat(),
            "score": bundle.score_column,
            "net": bundle.net_column,
            "family": bundle.ev_score_family_id,
            "geometry": bundle.geometry_bundle_sha256,
            "side_maps": sorted(bundle.side_maps),
            "producer_lineage": dict(bundle.producer_lineage),
            "spec": bundle.spec.__dict__,
        }),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bundle.manifest = dict(manifest)
    return manifest


def load_strict_r3_ev_bridge(directory: Path) -> StrictR3EVBridgeBundle:
    directory = Path(directory)
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != BRIDGE_SCHEMA:
        raise ValueError("not a strict-R3 common-bps EV bridge")
    payload = directory / str(manifest.get("bundle_file", ""))
    if not payload.is_file() or _file_hash(payload) != manifest.get("bundle_sha256"):
        raise ValueError("strict-R3 EV bridge artifact hash mismatch")
    bundle = joblib.load(payload)
    if not isinstance(bundle, StrictR3EVBridgeBundle):
        raise TypeError("strict-R3 EV bridge has the wrong payload type")
    bundle.__post_init__()
    if bundle.fit_cutoff.isoformat() != str(manifest.get("fit_cutoff")):
        raise ValueError("strict-R3 EV bridge cutoff manifest mismatch")
    if bundle.ev_score_family_id != str(manifest.get("ev_score_family_id")):
        raise ValueError("strict-R3 EV bridge score-family manifest mismatch")
    if bundle.geometry_bundle_sha256 != str(manifest.get("geometry_bundle_sha256")):
        raise ValueError("strict-R3 EV bridge geometry manifest mismatch")
    if dict(bundle.producer_lineage) != dict(manifest.get("producer_lineage", {})):
        raise ValueError("strict-R3 EV bridge producer-lineage manifest mismatch")
    bundle.manifest = dict(manifest)
    return bundle


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if not len(ordered):
        return float("nan")
    trim = int(np.floor(len(ordered) * trim_fraction))
    kept = ordered[trim:len(ordered) - trim] if len(ordered) > 2 * trim else ordered
    return float(kept.mean())


def _causal_residual_correction(
    frame: pd.DataFrame,
    *,
    residual_column: str,
    decision_column: str,
    label_available_column: str,
    spec: EVBridgeSpec,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, pd.DataFrame]:
    """Return broad-to-recent shrunk residual corrections without score pooling."""
    work = frame.copy()
    work["__bridge_original_position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values([decision_column, DEFAULT_IDENTITY_COLUMN], kind="stable").reset_index(drop=True)
    decision = pd.to_datetime(work[decision_column], utc=True, errors="raise")
    available = pd.to_datetime(work[label_available_column], utc=True, errors="raise")
    side = work["side_name"].astype(str).str.lower().to_numpy(object)
    residual = pd.to_numeric(work[residual_column], errors="coerce").to_numpy(float)
    rows = len(work)
    correction = np.zeros(rows, dtype=float)
    support = {window: np.zeros(rows, dtype=np.int64) for window in spec.residual_windows_days}
    status = np.full(rows, "bridge_prior_only_no_recent_residual_support", dtype=object)
    available_ns = available.array.as_unit("ns").asi8
    day = decision.dt.normalize()
    day_ns = day.array.as_unit("ns").asi8
    starts = np.r_[0, np.flatnonzero(day_ns[1:] != day_ns[:-1]) + 1] if rows else np.empty(0, dtype=np.int64)
    ends = np.r_[starts[1:], rows] if rows else np.empty(0, dtype=np.int64)
    indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    finite = np.isfinite(residual)
    for name in SIDES:
        positions = np.flatnonzero((side == name) & finite)
        ordered = positions[np.argsort(available_ns[positions], kind="stable")]
        indices[name] = available_ns[ordered], ordered
    audit: list[dict[str, object]] = []
    for start, end in zip(starts, ends, strict=True):
        snapshot = pd.Timestamp(day.iloc[start])
        snapshot_ns = int(day_ns[start])
        positions = np.arange(start, end, dtype=np.int64)
        for name in SIDES:
            current = positions[side[positions] == name]
            available_index, reference_positions = indices[name]
            estimate = 0.0
            any_support = False
            counts: dict[int, int] = {}
            max_available = pd.NaT
            # Broader history establishes the prior residual; later, shorter
            # windows are shrinkage updates rather than raw-score maps.
            for window, shrinkage in zip(
                spec.residual_windows_days, spec.residual_shrinkage_rows, strict=True,
            ):
                lower = int(np.searchsorted(
                    available_index,
                    snapshot_ns - pd.Timedelta(days=window).value,
                    side="left",
                ))
                upper = int(np.searchsorted(available_index, snapshot_ns, side="left"))
                reference = reference_positions[lower:upper]
                count = int(len(reference))
                counts[window] = count
                if count:
                    # Every broad-to-recent window ends at the same strict
                    # snapshot boundary, so this is also the exact latest
                    # resolved label that can affect this side/date estimate.
                    max_available = pd.Timestamp(available.iloc[reference[-1]])
                if count >= spec.minimum_residual_rows:
                    mean = _trimmed_mean(residual[reference], spec.residual_trim_fraction)
                    weight = count / (count + float(shrinkage))
                    estimate = weight * mean + (1.0 - weight) * estimate
                    any_support = True
                if len(current):
                    support[window][current] = count
            if len(current):
                correction[current] = estimate
                status[current] = (
                    "bridge_prior_plus_causal_residual"
                    if any_support else "bridge_prior_only_no_recent_residual_support"
                )
            audit.append({
                "snapshot_utc": snapshot,
                "side_name": name,
                "current_rows": int(len(current)),
                "residual_correction_bps": float(estimate),
                "residual_mapping_status": (
                    "mapped" if any_support else "prior_only_insufficient_residual_support"
                ),
                "strictly_prior_resolved": bool(max_available < snapshot) if pd.notna(max_available) else True,
                "reference_max_label_available_ts": max_available,
                **{f"residual_reference_rows_{window}d": counts[window] for window in spec.residual_windows_days},
            })
    if audit and not all(bool(item["strictly_prior_resolved"]) for item in audit):
        raise AssertionError("EV bridge residual correction consumed an unresolved outcome")
    # Restore caller order by an explicit position rather than relying on a
    # DataFrame index, which may be non-unique in a concatenated live ledger.
    ordered = pd.DataFrame({
        "__bridge_original_position__": work["__bridge_original_position__"].to_numpy(),
        "__correction__": correction,
        "__status__": status,
        **{f"__support_{window}__": value for window, value in support.items()},
    }).sort_values("__bridge_original_position__", kind="stable").reset_index(drop=True)
    return (
        ordered["__correction__"].to_numpy(float),
        {window: ordered[f"__support_{window}__"].to_numpy(np.int64) for window in spec.residual_windows_days},
        ordered["__status__"].to_numpy(object),
        pd.DataFrame(audit),
    )


def apply_strict_r3_ev_bridge(
    frame: pd.DataFrame,
    *,
    bundle: StrictR3EVBridgeBundle,
    net_column: str = DEFAULT_NET_COLUMN,
    decision_column: str = DEFAULT_DECISION_COLUMN,
    label_available_column: str = DEFAULT_LABEL_AVAILABLE_COLUMN,
    identity_column: str = DEFAULT_IDENTITY_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map a score/outcome ledger to executable common-bps expected value.

    Rows without an outcome (the live snapshot) receive the same frozen prior
    as resolved rows but cannot contribute to the causal residual correction.
    Thus a new upstream/conversion producer does not create an artificial
    no-trade day.  A row remains inadmissible only if its inputs are invalid or
    its *economic* expected net is below the fixed floor.
    """
    required = {
        identity_column, decision_column, "side_name", bundle.score_column, net_column,
        label_available_column, "ev_score_family_id", "geometry_bundle_sha256",
        "stack_is_prequential",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EV bridge admission frame lacks: {missing}")
    if identity_column != DEFAULT_IDENTITY_COLUMN:
        raise ValueError("strict-R3 EV bridge uses immutable candidate_id identities")
    _validate_score_semantics(
        frame,
        ev_score_family_id=bundle.ev_score_family_id,
        geometry_bundle_sha256=bundle.geometry_bundle_sha256,
        score_column=bundle.score_column,
        require_prequential=True,
    )
    # A common-bps bridge intentionally carries no producer restriction.  An
    # immediate reserve calibrator does: its score-to-bps map was fit on the
    # exact upstream × conversion producer which scored its reserve.  Enforce
    # that lineage at application time as well as in the replay join, so a
    # future caller cannot silently borrow a fresh producer's calibration.
    for column, expected in bundle.producer_lineage.items():
        if column not in frame.columns:
            raise ValueError(
                f"producer-specific EV bridge requires lineage column {column}"
            )
        if not frame[column].astype(str).eq(str(expected)).all():
            raise ValueError(
                f"producer-specific EV bridge rejects mismatched {column}"
            )
    out = frame.copy()
    out[decision_column] = pd.to_datetime(out[decision_column], utc=True, errors="raise")
    out[label_available_column] = pd.to_datetime(out[label_available_column], utc=True, errors="raise")
    if (out[label_available_column] <= out[decision_column]).any():
        raise ValueError("EV bridge requires label availability strictly after decision")
    out["ev_bridge_prior_expected_net_bps"] = bundle.predict_prior(out)
    realised = pd.to_numeric(out[net_column], errors="coerce").to_numpy(float)
    if "policy_path_valid" in out:
        realised = np.where(
            out["policy_path_valid"].fillna(False).astype(bool).to_numpy(),
            realised,
            np.nan,
        )
    out["ev_bridge_policy_residual_bps"] = realised - out["ev_bridge_prior_expected_net_bps"].to_numpy(float)
    correction, supports, status, audit = _causal_residual_correction(
        out,
        residual_column="ev_bridge_policy_residual_bps",
        decision_column=decision_column,
        label_available_column=label_available_column,
        spec=bundle.spec,
    )
    out["ev_bridge_recent_residual_bps"] = correction
    out["causal_21d_side_expected_net_bps"] = (
        out["ev_bridge_prior_expected_net_bps"].to_numpy(float) + correction
    )
    for window, values in supports.items():
        out[f"ev_bridge_residual_reference_rows_{window}d"] = values
    out["ev_bridge_residual_mapping_status"] = status
    out["causal_21d_side_mapping_status"] = status
    out["causal_21d_side_admitted_ge_50bps"] = (
        np.isfinite(out["causal_21d_side_expected_net_bps"])
        & out["causal_21d_side_expected_net_bps"].ge(bundle.spec.net_floor_bps)
    )
    out["ev_mapping_score_family_id"] = bundle.ev_score_family_id
    out["ev_mapping_geometry_bundle_sha256"] = bundle.geometry_bundle_sha256
    # Preserve the score producer actually being admitted.  The immediate
    # reserve mode requires these columns to be one exact pair; the legacy
    # common-bps bridge may contain multiple rows while it calculates only
    # bps residuals, but a live snapshot is still emitted with its own pair.
    if "conversion_bundle_sha256" in out:
        out["ev_mapping_conversion_vintage"] = out[
            "conversion_bundle_sha256"
        ].astype(str)
    if "upstream_bundle_sha256" in out:
        out["ev_mapping_upstream_vintage"] = out[
            "upstream_bundle_sha256"
        ].astype(str)
    out["ev_mapping_vintage_mode"] = bundle.calibration_mode
    out["ev_bridge_bundle_identity"] = str(bundle.manifest.get("bundle_identity", "unpersisted"))
    if out.loc[out[net_column].isna(), "causal_21d_side_admitted_ge_50bps"].isna().any():
        raise AssertionError("unresolved snapshot row has an indeterminate EV bridge admission")
    if not np.array_equal(
        out["causal_21d_side_admitted_ge_50bps"].to_numpy(bool),
        out["causal_21d_side_expected_net_bps"].ge(bundle.spec.net_floor_bps).fillna(False).to_numpy(bool),
    ):
        raise AssertionError("EV bridge admission must equal finite expected bps >= declared floor")
    audit["ev_mapping_vintage_mode"] = bundle.calibration_mode
    audit["ev_bridge_bundle_identity"] = str(bundle.manifest.get("bundle_identity", "unpersisted"))
    audit["ev_score_family_id"] = bundle.ev_score_family_id
    audit["geometry_bundle_sha256"] = bundle.geometry_bundle_sha256
    return out, audit


__all__ = [
    "BRIDGE_SCHEMA",
    "COMMON_BPS_CALIBRATION_MODE",
    "EXACT_PRODUCER_RESERVE_CALIBRATION_MODE",
    "EVBridgeSpec",
    "StrictR3EVBridgeBundle",
    "fit_strict_r3_ev_bridge",
    "persist_strict_r3_ev_bridge",
    "load_strict_r3_ev_bridge",
    "apply_strict_r3_ev_bridge",
]
