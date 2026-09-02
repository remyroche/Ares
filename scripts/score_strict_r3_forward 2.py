#!/usr/bin/env python3
"""Score prior-28-day and held candidates with one canonical bundle.

This command performs no fitting and consumes no outcomes.  Evaluation labels
must be joined only after its immutable prediction artifact is complete.  The
default is the current schema-v5 D2/conditional-consensus stack with a
timestamp-local top-30% base routing gate.
The legacy schema-v2 scorer remains opt-in for historical reconciliation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    CORRECTNESS_TRAIN_FRACTION,
    K9_TEMPERATURE_SCALE,
    LIVE_BASE_ROUTE_FRACTION,
    REFERENCE_DAYS,
    SCHEMA as CURRENT_SCHEMA,
    advance_lockstep_geometry_k9_state,
    initialize_lockstep_geometry_k9_state,
    load_lockstep_geometry_k9_state,
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    persist_lockstep_geometry_k9_state,
    score_four_week_conversion_by_upstream_vintage,
    score_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    SCHEMA as LEGACY_SCHEMA,
    assert_scoring_frame_is_target_free,
    load_monthly_bundle as load_legacy_bundle,
    score_same_model_reference as score_legacy_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _join(candidates_path: Path, features_path: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_path)
    features = pd.read_parquet(features_path)
    if "candidate_id" in features:
        feature_columns = [column for column in features.columns if column != "candidate_id"]
        result = candidates.merge(
            features[["candidate_id", *feature_columns]], on="candidate_id",
            how="left", validate="one_to_one", suffixes=("", "__feature"),
        )
    else:
        keys = ["__decision_ts__", "__symbol__", "side_name"]
        result = candidates.merge(features, on=keys, how="left", validate="one_to_one", suffixes=("", "__feature"))
    if result["candidate_id"].duplicated().any():
        raise ValueError("candidate-feature join changed identity uniqueness")
    return result


def _frozen_base_availability(
    frame: pd.DataFrame,
    fields: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Describe raw frozen-base availability without imputing a live row.

    The fitted boosters retain their historical median-imputation behaviour,
    but an input that lacks any declared base field is not eligible for a live
    trade.  Persisting this state alongside the score makes the distinction
    explicit and lets admission fail closed without deleting a point-in-time
    candidate or hiding its rejection reason.
    """
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(
        np.isfinite(values.to_numpy(dtype=float, copy=False)),
        index=values.index,
        columns=values.columns,
    )
    output = frame.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    output["frozen_base_feature_count"] = finite.sum(axis=1).astype("int16")
    output["frozen_base_feature_fraction"] = (
        finite.mean(axis=1).astype("float32")
    )
    output["frozen_base_contract_complete"] = finite.all(axis=1).astype(bool)
    return output


_K9_REPORTING_FIELDS = (
    "k9_entropy",
    "k9_top2_margin",
    "k9_ood_distance",
    "k9_cluster_weighted_fit_support_log",
    "k9_cluster_weighted_ood",
    "k9_cluster_timestamp_mahalanobis_train",
    "k9_cluster_timestamp_cov_break_train",
    "k9_cluster_timestamp_corr_break_train",
)


def _add_k9_cross_section_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Persist causal same-decision K9 cross-sectional percentiles.

    These are reporting-only fields, calculated after scoring over the full
    score-eligible timestamp population.  They never enter a model, mapper,
    admission gate, ranking, sizing, or exit decision.
    """
    output = frame.copy()
    groups = [output["__decision_ts__"], output["__score_role__"]]
    for field in _K9_REPORTING_FIELDS:
        if field not in output:
            continue
        values = pd.to_numeric(output[field], errors="coerce")
        grouped = values.groupby(groups, sort=False)
        output[f"{field}__xsec_percentile"] = grouped.rank(
            pct=True,
            ascending=True,
            method="average",
        ).astype("float32")
    return output


def _monthly_bundle_dir(root: Path, month: str) -> Path:
    candidates = (
        root / f"month={month}",
        root / f"cutoff={month.replace('-', '')}01",
        root / month,
    )
    for candidate in candidates:
        if (candidate / "run_manifest.json").exists():
            return candidate
    raise FileNotFoundError(f"no monthly upstream bundle for {month} under {root}")


def _score_current_by_upstream_vintage(
    *,
    conversion_bundle: object,
    reference: pd.DataFrame,
    held: pd.DataFrame,
    upstream_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Score each held upstream vintage against its own prior-score domain.

    A conversion score is only comparable with a CDF reference generated by
    the *same* fitted upstream base/consensus bundle.  The held block can
    cross a calendar-month upstream refit, so reference rows are deliberately
    rescored once per held upstream vintage.  Those duplicated reference rows
    are provenance, not candidates; held identities remain unique.

    ``conversion_bundle.cutoff - 28d`` can precede a later monthly upstream
    cutoff by more than 28 days.  It remains a valid target-free score
    reference: the conversion bundle itself defines its exact bounded window,
    and no labels are consumed while generating it.
    """
    work = held.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True)
    reference = reference.copy()
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    months = sorted(work["__decision_ts__"].dt.strftime("%Y-%m").unique().tolist())
    upstream_bundles = {
        str(month): load_monthly_upstream_bundle(
            _monthly_bundle_dir(upstream_root, str(month)),
        )
        for month in months
    }
    predictions, audit = score_four_week_conversion_by_upstream_vintage(
        conversion_bundle,
        reference=reference,
        held=work,
        upstream_bundles=upstream_bundles,
        route_top_fraction=LIVE_BASE_ROUTE_FRACTION,
    )
    hashes: dict[str, str] = {}
    for month, upstream_bundle in upstream_bundles.items():
        hashes[month] = str(upstream_bundle.manifest["bundle_sha256"])
    return predictions, audit, hashes


def _score_current_lockstep(
    *,
    conversion_bundle: object,
    upstream_bundle: object,
    reference: pd.DataFrame | None,
    held: pd.DataFrame,
    chunk_hours: int,
    geometry_k9_state_in: Path | None = None,
    geometry_k9_state_out: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Score the exact persisted upstream/conversion producer pair.

    The same pair scores the preceding 28-day target-free reserve before the
    held rows.  This is the forward counterpart of the lock-step historical
    producer and preserves the causal K9 history across score chunks.
    """
    conversion_cutoff = pd.Timestamp(conversion_bundle.cutoff)
    upstream_cutoff = pd.Timestamp(upstream_bundle.cutoff)
    if conversion_cutoff != upstream_cutoff:
        raise ValueError("lock-step forward scoring requires matching component cutoffs")
    if pd.Timestamp(conversion_bundle.end_exclusive) != pd.Timestamp(upstream_bundle.end_exclusive):
        raise ValueError("lock-step forward scoring requires matching component windows")
    held_start = pd.to_datetime(held["__decision_ts__"], utc=True).min()
    upstream_hash = str(upstream_bundle.manifest["bundle_sha256"])
    state_input_manifest: dict[str, object] | None = None
    if geometry_k9_state_in is None:
        # Bootstrap only: construct the immutable same-model reference and
        # score the complete activation-to-current held prefix once.  Recurring
        # hours must use the sealed state branch below.
        if reference is None:
            raise ValueError("lock-step state bootstrap requires the prior-28-day reference")
        expected_held_start = conversion_cutoff + pd.Timedelta(hours=1)
        if held_start != expected_held_start:
            raise ValueError(
                "lock-step forward scoring requires the exact continuous target-free "
                "held prefix beginning activation + 1h; an early or short-window "
                "start would change dynamic Geometry/K9 state"
            )
        reference_start = conversion_cutoff - pd.Timedelta(days=REFERENCE_DAYS)
        reference_score = score_monthly_upstream_bundle(
            upstream_bundle,
            reference,
            allow_prior_reference=True,
            prior_reference_start=reference_start,
            # Preserve the exact full-reference CDF coordinate used by frozen
            # MC1_d2. Candidate-specific held work still stops below timestamp
            # top-30, but calibration support may not be truncated after freeze.
            route_top_fraction=None,
        )
        reference_input = reference.merge(
            reference_score, on=["candidate_id", "__decision_ts__", "side_name"],
            how="left", validate="one_to_one",
        )
        state, scored_reference = initialize_lockstep_geometry_k9_state(
            conversion_bundle,
            reference=reference_input,
            upstream_bundle_sha256=upstream_hash,
            chunk_hours=chunk_hours,
        )
        require_single_next_hour = False
    else:
        state, state_input_manifest = load_lockstep_geometry_k9_state(
            geometry_k9_state_in,
            bundle=conversion_bundle,
            upstream_bundle_sha256=upstream_hash,
            expected_next_decision_ts=held_start,
        )
        scored_reference = pd.DataFrame()
        require_single_next_hour = True
    held_score = score_monthly_upstream_bundle(
        upstream_bundle, held, route_top_fraction=LIVE_BASE_ROUTE_FRACTION,
    )
    keys = ["candidate_id", "__decision_ts__", "side_name"]
    held_input = held.merge(held_score, on=keys, how="left", validate="one_to_one")
    base_fields = [
        "base_score", "base_rank42", "base_anchor_bps",
        "base_route_timestamp_top20", "base_route_status",
    ]
    downstream_fields = [
        "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream", "upstream_bundle_sha256",
    ]
    validation_frames = [("held", held_input)]
    if geometry_k9_state_in is None:
        validation_frames.insert(0, ("reference", reference_input))
    for role, frame in validation_frames:
        if frame.loc[:, base_fields].isna().any().any():
            raise AssertionError(f"lock-step {role} lost base-route coverage")
        routed = frame["base_route_timestamp_top20"].fillna(False).astype(bool)
        if frame.loc[routed, downstream_fields].isna().any().any():
            raise AssertionError(f"lock-step {role} lost routed upstream score coverage")
        if frame.loc[~routed, downstream_fields[:-1]].notna().any().any():
            raise AssertionError(f"lock-step {role} computed downstream scores below route")
    next_state, scored_held = advance_lockstep_geometry_k9_state(
        conversion_bundle,
        state=state,
        held=held_input,
        require_single_next_hour=require_single_next_hour,
    )
    if geometry_k9_state_out is not None:
        persisted = persist_lockstep_geometry_k9_state(
            next_state,
            out_dir=geometry_k9_state_out,
            predecessor_manifest_sha256=(
                str(state_input_manifest["manifest_sha256"])
                if state_input_manifest is not None else None
            ),
        )
    else:
        persisted = None
    predictions = pd.concat([scored_reference, scored_held], ignore_index=True)
    audit = pd.DataFrame([{
        "schema": CURRENT_SCHEMA,
        "cutoff": conversion_cutoff,
        "reference_rows": int(len(scored_reference) if geometry_k9_state_in is None else state.reference_rows),
        "held_rows": int(len(scored_held)),
        "score_chunk_hours": int(chunk_hours),
        "same_conversion_model_reference_and_held": True,
        "same_upstream_bundle_reference_and_held": True,
        "upstream_scores_are_prequential_lockstep": True,
        "held_percentile_operations": 0,
        "raw_k9_in_correctness": False,
        "geometry_refit_cadence": "never",
        "geometry_parent_bundle_sha256": conversion_bundle.geometry.parent_bundle_sha256,
        "geometry_dynamic_history": (
            "sealed target-free Geometry/K9 state strictly before current hour"
            if geometry_k9_state_in is not None else
            "frozen target-free history strictly before reference; then complete-universe chronological score chunks"
        ),
        "final_reference": state.final_reference.source,
        "memory_bound_complete_hour_chunks": True,
        "base_route_scope": "timestamp_local",
        "base_route_top_fraction": LIVE_BASE_ROUTE_FRACTION,
        "base_route_reference": "current_timestamp_base_score_cross_section",
        "below_route_candidate_models_skipped": True,
        "geometry_state_population": "complete_point_in_time_universe",
        "severe_affects_final_score": False,
        "geometry_k9_state_mode": (
            "restore_one_complete_hour" if geometry_k9_state_in is not None
            else "bootstrap_activation_to_current"
        ),
        "geometry_k9_state_input": (
            str(geometry_k9_state_in) if geometry_k9_state_in is not None else None
        ),
        "geometry_k9_state_input_manifest_sha256": (
            state_input_manifest.get("manifest_sha256") if state_input_manifest else None
        ),
        "geometry_k9_state_output_manifest_sha256": (
            persisted.get("manifest_sha256") if persisted else None
        ),
    }])
    audit["full_reference_for_frozen_mc1_coordinate"] = True
    audit["held_base_route_top_fraction"] = LIVE_BASE_ROUTE_FRACTION
    audit["same_upstream_bundle_for_reference_and_held"] = True
    audit["lockstep_producer"] = True
    return predictions, audit, {"lockstep": upstream_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--upstream-bundle-root",
        type=Path,
        help="Required by current-v5: directory containing month=YYYY-MM upstream bundles.",
    )
    parser.add_argument(
        "--upstream-bundle-dir",
        type=Path,
        help=(
            "Current-v5 exact lock-step upstream bundle. Mutually exclusive with "
            "--upstream-bundle-root."
        ),
    )
    parser.add_argument(
        "--lockstep-score-chunk-hours",
        type=int,
        default=72,
        help="Complete-hour chunk size for exact lock-step K9-state scoring.",
    )
    parser.add_argument(
        "--lockstep-geometry-k9-state-in",
        type=Path,
        help=(
            "Exact immutable Geometry/K9 state from the immediately preceding "
            "successful lock-step hour. When supplied, score exactly one new "
            "complete timestamp and never replay the activation prefix."
        ),
    )
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--held-candidates", type=Path, required=True)
    parser.add_argument("--held-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--schema",
        choices=("current-v5", "legacy-v2"),
        default="current-v5",
        help="Production defaults to frozen-geometry current-v5; legacy-v2 is reconciliation-only.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable score output already exists: {args.out_dir}")
    if args.schema == "current-v5":
        if (args.upstream_bundle_root is None) == (args.upstream_bundle_dir is None):
            parser.error(
                "current-v5 requires exactly one of --upstream-bundle-root "
                "(staggered) or --upstream-bundle-dir (exact lock-step)"
            )
        if args.lockstep_score_chunk_hours < 1:
            parser.error("--lockstep-score-chunk-hours must be positive")
        if args.lockstep_geometry_k9_state_in is not None and args.upstream_bundle_dir is None:
            parser.error("Geometry/K9 state restore requires --upstream-bundle-dir exact lock-step mode")
        bundle = load_four_week_conversion_bundle(args.bundle_dir)
        schema = CURRENT_SCHEMA
        selected = [
            "candidate_id", "__decision_ts__", "__symbol__", "side_name",
            *bundle.base_fields,
        ]
    else:
        bundle = load_legacy_bundle(args.bundle_dir)
        schema = LEGACY_SCHEMA
        score = score_legacy_bundle
        selected = [
            "candidate_id", "__decision_ts__", "__symbol__", "side_name",
            *bundle.base_fields, *bundle.context_fields,
        ]
    held = _join(args.held_candidates, args.held_features)
    held_availability = _frozen_base_availability(held, bundle.base_fields)
    held = held.loc[:, list(dict.fromkeys(selected))]
    assert_scoring_frame_is_target_free(held)
    # The state contains the immutable, same-model CDF built from the
    # preceding 28-day reserve.  Do not read/re-score that reserve on a
    # recurring hour: its state hashes are validated before the current hour
    # reaches a model. Bootstrap remains explicit and target-free.
    if args.lockstep_geometry_k9_state_in is None:
        reference = _join(args.reference_candidates, args.reference_features)
        reference_availability = _frozen_base_availability(reference, bundle.base_fields)
        reference = reference.loc[:, list(dict.fromkeys(selected))]
        assert_scoring_frame_is_target_free(reference)
    else:
        reference = None
        reference_availability = pd.DataFrame(columns=[
            "candidate_id", "__decision_ts__", "side_name",
            "frozen_base_feature_count", "frozen_base_feature_fraction",
            "frozen_base_contract_complete",
        ])
    upstream_hashes: dict[str, str] = {}
    producer_topology = "legacy"
    if args.schema == "current-v5":
        if args.upstream_bundle_dir is not None:
            upstream = load_monthly_upstream_bundle(args.upstream_bundle_dir)
            predictions, audit, upstream_hashes = _score_current_lockstep(
                conversion_bundle=bundle,
                upstream_bundle=upstream,
                reference=reference,
                held=held,
                chunk_hours=args.lockstep_score_chunk_hours,
                geometry_k9_state_in=args.lockstep_geometry_k9_state_in,
                geometry_k9_state_out=(args.out_dir / "geometry_k9_state"),
            )
            producer_topology = "exact_lockstep_shared_cutoff"
        else:
            predictions, audit, upstream_hashes = _score_current_by_upstream_vintage(
                conversion_bundle=bundle,
                reference=reference,
                held=held,
                upstream_root=args.upstream_bundle_root,
            )
            producer_topology = "staggered_monthly_upstream"
        # Stable aliases are required by the canonical post-admission LDF
        # sidecar.  They preserve the producer's monthly prequential score
        # semantics and are available for reference and held rows alike.
        predictions["base_rank"] = pd.to_numeric(
            predictions["base_rank42"], errors="coerce",
        )
        predictions["consensus_rank"] = pd.to_numeric(
            predictions["conditional_consensus_rank"], errors="coerce",
        )
        predictions["stack_is_prequential"] = True
        # This is the producer activation boundary, not an outcome-derived
        # field.  It keys the exact, pre-fit 28-day reserve EV map used by
        # forward admission, so a later refit cannot accidentally reuse an
        # older producer's calibration artifact.
        predictions["calibration_activation_ts"] = pd.Timestamp(
            bundle.cutoff,
        ).tz_convert("UTC")
    else:
        predictions, audit = score(bundle, reference=reference, held=held)
    availability_parts = [held_availability.assign(__score_role__="held")]
    if not reference_availability.empty:
        availability_parts.insert(0, reference_availability.assign(__score_role__="reference"))
    availability = pd.concat(availability_parts, ignore_index=True)
    predictions = predictions.merge(
        availability,
        on=["candidate_id", "__decision_ts__", "side_name", "__score_role__"],
        how="left",
        # A bounded prior-reference is intentionally repeated once for
        # each held upstream vintage.  Only the held output is the candidate
        # prediction population, and it remains unique.
        validate="many_to_one",
    )
    if predictions["frozen_base_contract_complete"].isna().any():
        raise AssertionError("forward score lost frozen-base availability lineage")
    predictions = _add_k9_cross_section_reference(predictions)
    # The lockstep Geometry/K9 state is persisted before the tabular score
    # artifacts and therefore may already have created the parent directory.
    # Immutability is enforced by the explicit preflight check above.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.out_dir / "score_decomposition.parquet", index=False, compression="zstd")
    predictions.loc[predictions["__score_role__"].eq("held")].drop(columns="__score_role__").to_parquet(
        args.out_dir / "predictions.parquet", index=False, compression="zstd",
    )
    predictions.loc[predictions["__score_role__"].eq("reference")].drop(columns="__score_role__").to_parquet(
        args.out_dir / "same_model_prior42_reference_scores.parquet", index=False, compression="zstd",
    )
    audit.to_parquet(args.out_dir / "same_model_reference_replay_audit.parquet", index=False)
    manifest = {
        "schema": f"{schema}_score_output",
        "producer_contract": args.schema,
        "bundle_dir": str(args.bundle_dir),
        "bundle_sha256": bundle.manifest["bundle_sha256"],
        "calibration_activation_ts": (
            pd.Timestamp(bundle.cutoff).tz_convert("UTC").isoformat()
            if args.schema == "current-v5" else None
        ),
        "geometry_bundle_sha256": bundle.geometry.bundle_sha256,
        "geometry_contract": (
            "one_frozen_oct_dec_2024_geometry_K9_view_temperature_0.25"
            if args.schema == "current-v5"
            else "fit_once_2024-10-01_to_2025-01-01_then_frozen"
        ),
        "reference_rows": int(
            predictions["__score_role__"].eq("reference").sum()
            if args.lockstep_geometry_k9_state_in is None
            else int(audit["reference_rows"].iloc[0])
        ),
        "reference_window_days": (
            REFERENCE_DAYS if args.schema == "current-v5" else 42
        ),
        "reference_file_wire_alias": "same_model_prior42_reference_scores.parquet",
        "reference_unique_candidate_rows": int(
            predictions.loc[predictions["__score_role__"].eq("reference"), "candidate_id"].nunique()
            if args.lockstep_geometry_k9_state_in is None
            else int(audit["reference_rows"].iloc[0])
        ),
        "held_rows": int(predictions["__score_role__"].eq("held").sum()),
        "held_complete_base_contract_rows": int(held_availability["frozen_base_contract_complete"].sum()),
        "held_complete_base_contract_fraction": float(held_availability["frozen_base_contract_complete"].mean()),
        "reference_complete_base_contract_rows": (
            int(reference_availability["frozen_base_contract_complete"].sum())
            if not reference_availability.empty else None
        ),
        "reference_complete_base_contract_fraction": (
            float(reference_availability["frozen_base_contract_complete"].mean())
            if not reference_availability.empty else None
        ),
        "outcome_columns_consumed": [], "held_percentile_operations": 0,
        "producer_topology": producer_topology,
        "same_bundle_for_reference_and_held": (
            args.schema == "legacy-v2" or producer_topology == "exact_lockstep_shared_cutoff"
        ),
        "same_conversion_bundle_for_reference_and_held": True,
        "same_upstream_bundle_for_reference_and_held_per_producer": (
            args.schema == "current-v5"
        ),
        "reference_upstream_rescored_by_conversion_cutoff_month_bundle": (
            producer_topology == "exact_lockstep_shared_cutoff"
        ),
        "reference_upstream_rescored_by_held_month_bundle": (
            args.schema == "current-v5" and producer_topology == "staggered_monthly_upstream"
        ),
        "reference_upstream_month": None,
        "held_upstream_bundle_is_selected_by_candidate_month": (
            producer_topology == "staggered_monthly_upstream"
        ),
        "monthly_upstream_bundle_hashes": upstream_hashes,
        "ev_score_family_ids": sorted(
            predictions["ev_score_family_id"].dropna().astype(str).unique().tolist()
        ) if "ev_score_family_id" in predictions else [],
        "conversion_bundle_vintages": sorted(
            predictions["conversion_bundle_sha256"].dropna().astype(str).unique().tolist()
        ) if "conversion_bundle_sha256" in predictions else [],
        "upstream_bundle_vintages": sorted(
            predictions["upstream_bundle_sha256"].dropna().astype(str).unique().tolist()
        ) if "upstream_bundle_sha256" in predictions else [],
        "canonical_consensus": (
            "conditional_usefulness_ten_head_v1"
            if args.schema == "current-v5"
            else "legacy_ordinary_equal_cap_heads"
        ),
        "base_route": (
            {
                "scope": "timestamp_local",
                "top_fraction": LIVE_BASE_ROUTE_FRACTION,
                "score": "base_score",
                "tie_break": "candidate_id_ascending",
                "shared_geometry_population": "complete_point_in_time_universe",
                "below_route": "stop_after_base_and_fail_closed",
            }
            if args.schema == "current-v5" else None
        ),
        "severe_affects_final_score": (
            False if args.schema == "current-v5" else True
        ),
        "correctness_training_fraction": (
            CORRECTNESS_TRAIN_FRACTION if args.schema == "current-v5" else None
        ),
        "correctness_gate_domain": (
            "pooled-global training upstream score only"
            if args.schema == "current-v5" else None
        ),
        "k9_temperature_scale": (
            K9_TEMPERATURE_SCALE if args.schema == "current-v5" else None
        ),
        "ldf_score_aliases": (
            {
                "base_rank": "base_rank42",
                "consensus_rank": "conditional_consensus_rank",
                "stack_is_prequential": True,
            }
            if args.schema == "current-v5" else None
        ),
        "source_hashes": {
            "reference_candidates": _sha(args.reference_candidates),
            "reference_features": _sha(args.reference_features),
            "held_candidates": _sha(args.held_candidates),
            "held_features": _sha(args.held_features),
        },
        "geometry_k9_state": {
            "mode": (
                "restore_one_complete_hour"
                if args.lockstep_geometry_k9_state_in is not None
                else "bootstrap_activation_to_current"
            ),
            "input": (
                str(args.lockstep_geometry_k9_state_in)
                if args.lockstep_geometry_k9_state_in is not None else None
            ),
            "output": str(args.out_dir / "geometry_k9_state"),
            "output_manifest_sha256": audit[
                "geometry_k9_state_output_manifest_sha256"
            ].iloc[0] if "geometry_k9_state_output_manifest_sha256" in audit else None,
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
