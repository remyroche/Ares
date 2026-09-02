#!/usr/bin/env python3
"""Apply the canonical causal EV admission map to target-free live scores.

The current score has no outcome.  This command combines it with an immutable
ledger of earlier, resolved policy outcomes only, then applies the exact
producer's equal-day 28-day map with symmetric 15% day trimming.  It is deliberately
separate from scoring so live decision inputs and resolved-outcome history are
independently hashable and auditable.
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
    CALIBRATION_RESERVE_DAYS,
    OptimizedPolicyContract,
    SCHEMA,
    _current_ev_score_family_id,
    apply_current_admission_snapshot,
)
from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
    load_strict_r3_ev_bridge,
)
from extreme_price_movements.strict_r3_cell_day_admission import (  # noqa: E402
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
    apply_cell_day_trim15_admission_snapshot,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    assert_scoring_frame_is_target_free,
)


IMMEDIATE_CALIBRATION_KEYS = (
    "ev_score_family_id",
    "geometry_bundle_sha256",
    "conversion_bundle_sha256",
    "upstream_bundle_sha256",
    "calibration_activation_ts",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_score_manifest(predictions: Path) -> dict[str, object]:
    path = predictions.parent / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"target-free score manifest is required: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("producer_contract") != "current-v5":
        raise ValueError("admission snapshot requires a current-v5 forward-score producer")
    if manifest.get("held_percentile_operations") != 0:
        raise ValueError("admission snapshot rejects held-window percentile scoring")
    if manifest.get("outcome_columns_consumed") not in ([], None):
        raise ValueError("admission snapshot rejects scores produced with outcomes")
    if not manifest.get("same_upstream_bundle_for_reference_and_held_per_producer", False):
        raise ValueError(
            "admission snapshot requires a same-upstream prior-score reference "
            "for every held producer"
        )
    return manifest


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _resolve_bundle_path(path: object) -> Path:
    """Resolve artifact paths recorded relative to the Ares repository."""
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_immediate_exact_reserve_calibrator(
    *,
    index_path: Path,
    current: pd.DataFrame,
    score_manifest: dict[str, object],
    decision_ts: pd.Timestamp,
) -> tuple[object, dict[str, object]]:
    """Load the one exact-producer reserve map valid for this live snapshot.

    A refit bundle must begin with a map created from its own excluded,
    resolved reserve.  This resolver deliberately has no nearest-vintage,
    score-family, or common-bps fallback: an absent or mismatched reserve map
    is a fail-closed deployment error.
    """
    if score_manifest.get("producer_topology") != "exact_lockstep_shared_cutoff":
        raise ValueError(
            "immediate exact-reserve admission requires an exact lock-step "
            "forward-score producer",
        )
    required_current = set(IMMEDIATE_CALIBRATION_KEYS)
    missing = sorted(required_current.difference(current.columns))
    if missing:
        raise ValueError(
            "current score ledger lacks immediate-calibration identity fields: "
            f"{missing}",
        )
    work = current.loc[:, list(IMMEDIATE_CALIBRATION_KEYS)].copy()
    work["calibration_activation_ts"] = pd.to_datetime(
        work["calibration_activation_ts"], utc=True, errors="raise",
    )
    if work.isna().any().any():
        raise ValueError("current score ledger has null immediate-calibration identities")
    if len(work.drop_duplicates()) != 1:
        raise ValueError("one live admission snapshot must have one exact producer identity")
    current_identity = work.drop_duplicates().iloc[0]
    activation = _utc(current_identity["calibration_activation_ts"])
    if activation > decision_ts:
        raise ValueError("current producer activation is after the live decision")

    index = pd.read_parquet(index_path)
    required_index = {
        *IMMEDIATE_CALIBRATION_KEYS,
        "producer_bundle_id", "status", "ev_bridge_bundle", "ev_bridge_bundle_sha256",
        "reference_min_decision_ts", "reference_max_decision_ts",
        "reference_max_label_available_ts",
    }
    missing = sorted(required_index.difference(index.columns))
    if missing:
        raise ValueError(f"immediate calibration index lacks: {missing}")
    index = index.copy()
    index["calibration_activation_ts"] = pd.to_datetime(
        index["calibration_activation_ts"], utc=True, errors="raise",
    )
    if index.duplicated(list(IMMEDIATE_CALIBRATION_KEYS)).any():
        raise ValueError("immediate calibration index has duplicate producer identities")
    match = index.loc[
        index["ev_score_family_id"].astype(str).eq(str(current_identity["ev_score_family_id"]))
        & index["geometry_bundle_sha256"].astype(str).eq(str(current_identity["geometry_bundle_sha256"]))
        & index["conversion_bundle_sha256"].astype(str).eq(str(current_identity["conversion_bundle_sha256"]))
        & index["upstream_bundle_sha256"].astype(str).eq(str(current_identity["upstream_bundle_sha256"]))
        & index["calibration_activation_ts"].eq(activation)
    ].copy()
    if len(match) != 1:
        raise ValueError(
            "no exact same-producer immediate calibration reserve exists for "
            "the live score bundle; fail closed rather than bridge vintages",
        )
    record = match.iloc[0]
    if record["status"] != "fitted_immediate_exact_producer_calibration":
        raise ValueError(
            "the exact producer immediate calibration reserve is not fitted: "
            f"{record['status']}: {record.get('error')}",
        )
    if pd.isna(record["ev_bridge_bundle"]):
        raise ValueError("fitted immediate calibration record has no EV-map artifact")
    reference_min = _utc(record["reference_min_decision_ts"])
    reference_max = _utc(record["reference_max_decision_ts"])
    reference_max_label = _utc(record["reference_max_label_available_ts"])
    reserve_start = activation - pd.Timedelta(days=CALIBRATION_RESERVE_DAYS)
    if reference_min < reserve_start or reference_max >= activation:
        raise ValueError("immediate calibration index violates the exact prior-28-day window")
    if reference_max_label >= activation:
        raise ValueError("immediate calibration index contains an unresolved activation-time label")
    bundle = load_strict_r3_ev_bridge(_resolve_bundle_path(record["ev_bridge_bundle"]))
    if bundle.calibration_mode != EXACT_PRODUCER_RESERVE_CALIBRATION_MODE:
        raise ValueError("immediate calibration artifact is not an exact-producer reserve map")
    if str(bundle.manifest.get("bundle_sha256")) != str(record["ev_bridge_bundle_sha256"]):
        raise ValueError("immediate calibration index and EV-map artifact hash differ")
    if _utc(bundle.fit_cutoff) != activation:
        raise ValueError("immediate calibration activation and map cutoff differ")
    expected_lineage = {
        "conversion_bundle_sha256": str(current_identity["conversion_bundle_sha256"]),
        "upstream_bundle_sha256": str(current_identity["upstream_bundle_sha256"]),
    }
    if dict(bundle.producer_lineage) != expected_lineage:
        raise ValueError("immediate calibration map has the wrong exact producer lineage")
    if bundle.ev_score_family_id != str(current_identity["ev_score_family_id"]):
        raise ValueError("immediate calibration map has the wrong score-family contract")
    if bundle.geometry_bundle_sha256 != str(current_identity["geometry_bundle_sha256"]):
        raise ValueError("immediate calibration map has the wrong frozen geometry contract")
    return bundle, {
        "immediate_calibration_index": str(index_path),
        "immediate_calibration_index_sha256": _sha(index_path),
        "immediate_calibration_producer_bundle_id": str(record["producer_bundle_id"]),
        "immediate_calibration_activation_ts": activation.isoformat(),
        "immediate_calibration_reference_window_days": CALIBRATION_RESERVE_DAYS,
        "immediate_calibration_reference_min_decision_ts": reference_min.isoformat(),
        "immediate_calibration_reference_max_decision_ts": reference_max.isoformat(),
        "immediate_calibration_reference_max_label_available_ts": reference_max_label.isoformat(),
        "immediate_calibration_ev_bridge_bundle": str(record["ev_bridge_bundle"]),
        "immediate_calibration_ev_bridge_bundle_sha256": str(record["ev_bridge_bundle_sha256"]),
    }


def _verify_policy_lineage(
    *, resolved_ledger: Path, policy_json: Path,
) -> dict[str, object]:
    """Reject EV-reference labels made under another exit policy or cost rule."""
    payload = json.loads(policy_json.read_text())
    winner = payload.get("winner", {})
    expected = OptimizedPolicyContract()
    observed = {
        "sl_mult": float(winner.get("sl_mult", np.nan)),
        "trailing_activation_mult": float(winner.get("trailing_activation_mult", np.nan)),
        "fixed_trailing_gap_mult": float(winner.get("fixed_trailing_gap_mult", np.nan)),
    }
    canonical = {
        "sl_mult": expected.stop_loss_atr,
        "trailing_activation_mult": expected.trailing_activation_atr,
        "fixed_trailing_gap_mult": expected.trailing_giveback_atr,
    }
    if any(not np.isclose(observed[key], canonical[key]) for key in canonical):
        raise ValueError("forward admission requires the frozen selected-policy geometry")

    source_manifest_path = resolved_ledger.parent / "run_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"resolved-policy lineage manifest is required: {source_manifest_path}")
    source = json.loads(source_manifest_path.read_text())
    declared_hash = source.get("policy_json_sha256")
    # Current lock-step producers record their canonical policy-outcome
    # ledger under ``policy_training_supervision``; earlier producers used
    # the top-level key.  Both are immutable provenance, so accept either
    # spelling but never continue without resolving its policy manifest.
    policy_outcomes = source.get("policy_outcomes")
    if policy_outcomes is None:
        policy_outcomes = (
            source.get("policy_training_supervision", {})
            .get("outcome_path")
        )
    if declared_hash is None and policy_outcomes is not None:
        outcome_path = Path(str(policy_outcomes))
        if not outcome_path.is_absolute():
            outcome_path = ROOT / outcome_path
        outcome_manifest = outcome_path.parent / "run_manifest.json"
        if not outcome_manifest.exists():
            raise FileNotFoundError(f"policy-outcome manifest is required: {outcome_manifest}")
        declared_hash = json.loads(outcome_manifest.read_text()).get("policy_json_sha256")
    actual_hash = _sha(policy_json)
    if declared_hash != actual_hash:
        raise ValueError("resolved policy outcomes do not match the declared frozen policy JSON")
    return {
        "policy_json": str(policy_json),
        "policy_json_sha256": actual_hash,
        "policy": canonical,
        "timeout_hours": expected.timeout_hours,
        "cost_bps_once": expected.cost_bps_once,
    }


def _attach_resolved_producer_lineage(
    resolved: pd.DataFrame,
    lineage_path: Path | None,
) -> pd.DataFrame:
    """Attach an immutable sidecar only when legacy rows lack producer IDs."""
    if "upstream_bundle_sha256" in resolved:
        return resolved
    if lineage_path is None:
        raise ValueError(
            "resolved OOF ledger lacks upstream_bundle_sha256; provide "
            "--resolved-producer-lineage rather than mixing producer vintages",
        )
    lineage = pd.read_parquet(lineage_path)
    required = {
        "candidate_id", "__decision_ts__", "conversion_bundle_sha256",
        "geometry_bundle_sha256", "upstream_bundle_sha256",
        "ev_score_family_id",
    }
    missing = sorted(required.difference(lineage.columns))
    if missing:
        raise ValueError(f"resolved producer lineage lacks: {missing}")
    if lineage["candidate_id"].duplicated().any():
        raise ValueError("resolved producer lineage has duplicate candidate IDs")
    merged = resolved.merge(
        lineage.loc[:, sorted(required)], on="candidate_id", how="left",
        validate="one_to_one", suffixes=("", "__lineage"),
    )
    if merged["upstream_bundle_sha256"].isna().any():
        raise ValueError("resolved producer lineage does not cover every usable OOF row")
    for column in (
        "__decision_ts__", "conversion_bundle_sha256", "geometry_bundle_sha256",
    ):
        if column in resolved:
            source = merged[column]
            sidecar = merged[f"{column}__lineage"]
            if not source.astype(str).eq(sidecar.astype(str)).all():
                raise ValueError(f"resolved producer lineage conflicts on {column}")
            merged = merged.drop(columns=f"{column}__lineage")
    if "ev_score_family_id" in resolved:
        if not merged["ev_score_family_id"].astype(str).eq(
            merged["ev_score_family_id__lineage"].astype(str),
        ).all():
            raise ValueError("resolved producer lineage conflicts on ev_score_family_id")
        merged = merged.drop(columns="ev_score_family_id__lineage")
    else:
        merged["ev_score_family_id"] = merged.pop("ev_score_family_id__lineage")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-score-label-ledger", type=Path, required=True)
    parser.add_argument(
        "--resolved-producer-lineage", type=Path,
        help=(
            "Required for older OOF ledgers without per-row upstream producer "
            "hashes; generated by attach_strict_r3_producer_lineage.py."
        ),
    )
    parser.add_argument("--current-predictions", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--ev-bridge-bundle", type=Path,
        help=(
            "Immutable strict-OOF common-bps bridge. When supplied, a new "
            "producer uses the frozen bridge prior immediately and a causal "
            "21/42/84-day policy-residual correction; raw scores remain "
            "unpooled across producer vintages. Omit only for the exact-"
            "producer fail-closed audit control."
        ),
    )
    parser.add_argument(
        "--immediate-calibration-index", type=Path,
        help=(
            "Immutable exact-producer 28-day reserve-calibration index. "
            "Canonical forward admission requires this path and rejects a "
            "different producer, geometry, or activation identity."
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-admission-audit", action="store_true",
        help=(
            "Permit the legacy bridge/fail-closed controls for an explicit "
            "research audit. It must not be used for canonical exact-lockstep "
            "forward admission."
        ),
    )
    parser.add_argument(
        "--decision-ts", required=True,
        help="One UTC decision timestamp to admit; live admission is an hourly snapshot.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable admission output already exists: {args.out_dir}")
    if args.ev_bridge_bundle is not None and args.immediate_calibration_index is not None:
        parser.error(
            "--ev-bridge-bundle and --immediate-calibration-index are mutually exclusive",
        )

    score_manifest = _load_score_manifest(args.current_predictions)
    if (
        score_manifest.get("producer_topology") == "exact_lockstep_shared_cutoff"
        and args.immediate_calibration_index is None
        and not args.allow_noncanonical_admission_audit
    ):
        raise ValueError(
            "canonical exact-lockstep forward admission requires "
            "--immediate-calibration-index; use --allow-noncanonical-admission-audit "
            "only for a labelled research control",
        )
    policy_lineage = _verify_policy_lineage(
        resolved_ledger=args.resolved_score_label_ledger,
        policy_json=args.policy_json,
    )
    resolved = pd.read_parquet(args.resolved_score_label_ledger)
    current = pd.read_parquet(args.current_predictions)
    assert_scoring_frame_is_target_free(current)
    required_resolved = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "policy_net_bps", "policy_label_available_ts", "geometry_bundle_sha256",
        "conversion_bundle_sha256", "stack_is_prequential",
    }
    missing = sorted(required_resolved.difference(resolved.columns))
    if missing:
        raise ValueError(f"resolved score ledger lacks: {missing}")
    required_current = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256",
        "ev_score_family_id", "stack_is_prequential",
        "frozen_base_contract_complete",
        "base_route_timestamp_top20", "base_route_status",
    }
    missing = sorted(required_current.difference(current.columns))
    if missing:
        raise ValueError(f"current score ledger lacks: {missing}")
    decision_ts = pd.to_datetime(args.decision_ts, utc=True)
    current["__decision_ts__"] = pd.to_datetime(current["__decision_ts__"], utc=True, errors="raise")
    current = current.loc[current["__decision_ts__"].eq(decision_ts)].copy()
    if current.empty:
        raise ValueError(f"current score ledger has no rows at {decision_ts.isoformat()}")
    # New score producers persist this identity per row.  Allow a score file
    # created by the immediately preceding scorer revision only when its own
    # immutable manifest declares the same conversion activation boundary.
    if (
        args.immediate_calibration_index is not None
        and "calibration_activation_ts" not in current.columns
    ):
        declared_activation = score_manifest.get("calibration_activation_ts")
        if declared_activation is None:
            raise ValueError(
                "exact-reserve admission requires calibration_activation_ts "
                "in the score ledger or its immutable score manifest",
            )
        current["calibration_activation_ts"] = _utc(declared_activation)
    resolved["policy_label_available_ts"] = pd.to_datetime(
        resolved["policy_label_available_ts"], utc=True, errors="raise",
    )
    # A live snapshot can consume only exact outcomes that had resolved before
    # this decision.  Filtering here also prevents a historical batch ledger
    # from accidentally supplying the current candidates' later outcomes.
    resolved = resolved.loc[
        resolved["policy_label_available_ts"].lt(decision_ts)
    ].copy()
    # The current producer persistently emits this deterministic semantic
    # family.  Older v5 OOF ledgers predate the field but already contain the
    # frozen geometry identity from which it is derived; attach it without
    # touching scores, labels or their timestamps so they remain replayable
    # under the repaired admission contract.
    if "ev_score_family_id" not in resolved:
        resolved["ev_score_family_id"] = resolved[
            "geometry_bundle_sha256"
        ].astype(str).map(_current_ev_score_family_id)
    resolved = _attach_resolved_producer_lineage(
        resolved, args.resolved_producer_lineage,
    )
    current_domain = current.loc[:, [
        "ev_score_family_id", "conversion_bundle_sha256",
        "upstream_bundle_sha256", "geometry_bundle_sha256",
    ]].drop_duplicates()
    if len(current_domain) != 1:
        raise ValueError("forward admission expects one current score-domain vintage")
    family, conversion, upstream, geometry = current_domain.iloc[0].astype(str).tolist()
    compatible_resolved_rows = int((
        resolved["ev_score_family_id"].astype(str).eq(family)
        & resolved["conversion_bundle_sha256"].astype(str).eq(conversion)
        & resolved["upstream_bundle_sha256"].astype(str).eq(upstream)
        & resolved["geometry_bundle_sha256"].astype(str).eq(geometry)
    ).sum())
    current_geometry = set(current["geometry_bundle_sha256"].dropna().astype(str))
    resolved_geometry = set(resolved["geometry_bundle_sha256"].dropna().astype(str))
    if len(current_geometry) != 1:
        raise ValueError("current admission snapshot does not have one frozen geometry identity")
    # At the first hour of an exact-reserve producer there may intentionally
    # be no post-activation resolved row at all.  The matching frozen geometry
    # and its reserve support are validated by the immediate-calibration
    # artifact below, so requiring an older live ledger row would re-create
    # the cold-start drought.  Legacy bridge/control modes still require a
    # shared historical geometry identity here.
    if (
        args.immediate_calibration_index is None
        and not current_geometry.issubset(resolved_geometry)
    ):
        raise ValueError("current and resolved admission ledgers do not share one frozen geometry identity")

    immediate_calibration: dict[str, object] | None = None
    if args.immediate_calibration_index is not None:
        bridge_bundle, immediate_calibration = _load_immediate_exact_reserve_calibrator(
            index_path=args.immediate_calibration_index,
            current=current,
            score_manifest=score_manifest,
            decision_ts=decision_ts,
        )
    else:
        bridge_bundle = (
            load_strict_r3_ev_bridge(args.ev_bridge_bundle)
            if args.ev_bridge_bundle is not None else None
        )
    if immediate_calibration is not None:
        admitted, audit = apply_cell_day_trim15_admission_snapshot(
            resolved_score_ledger=resolved,
            current_scores=current,
            bundle=bridge_bundle,
        )
    else:
        admitted, audit = apply_current_admission_snapshot(
            resolved_score_ledger=resolved,
            current_scores=current,
            ev_bridge_bundle=bridge_bundle,
        )
    incomplete = ~admitted["frozen_base_contract_complete"].fillna(False).astype(bool)
    below_base_route = ~admitted["base_route_timestamp_top20"].fillna(False).astype(bool)
    admitted["admission_rejection_reason"] = np.where(
        incomplete,
        "frozen_base_contract_incomplete",
        np.where(below_base_route, "stopped_after_base_below_timestamp_top20", ""),
    )
    # A score is retained for diagnosis, but no row with an unavailable frozen
    # input is permitted through the executable EV-admission gate.
    admitted.loc[incomplete, "causal_21d_side_admitted_ge_50bps"] = False
    admitted.loc[below_base_route, "causal_21d_side_admitted_ge_50bps"] = False
    args.out_dir.mkdir(parents=True)
    admitted.to_parquet(args.out_dir / "admitted_predictions.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "causal_21d_admission_audit.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_forward_admission_snapshot",
        "current_predictions": str(args.current_predictions),
        "current_predictions_sha256": _sha(args.current_predictions),
        "current_score_manifest_sha256": _sha(args.current_predictions.parent / "run_manifest.json"),
        "resolved_score_label_ledger": str(args.resolved_score_label_ledger),
        "resolved_score_label_ledger_sha256": _sha(args.resolved_score_label_ledger),
        "score_producer_contract": score_manifest["producer_contract"],
        "score_held_percentile_operations": 0,
        "current_outcomes_consumed": [],
        "policy_lineage": policy_lineage,
        "admission": (
            "same exact-producer fixed score cells; one equal-weight policy-net "
            "mean per UTC day x cell over 28 days; symmetric 15% day trimming; "
            "monotone expected-net curve >= +50 bps"
            if immediate_calibration is not None else (
                "frozen strict-OOF common-bps EV bridge plus prior-resolved "
                "21/42/84-day side-local policy-residual correction >= +50 bps"
                if bridge_bundle is not None
                else "prior-resolved hierarchical 21/42/84-day side-local EV map >= +50 bps; fail closed"
            )
        ),
        "ev_mapping_vintage_mode": (
            CELL_DAY_TRIM_15_CALIBRATION_MODE
            if immediate_calibration is not None else (
                "strict_oof_common_bps_bridge_plus_causal_residual_v1"
                if bridge_bundle is not None
                else "strict_full_producer_vintage_fail_closed_v2"
            )
        ),
        "ev_bridge_bundle": (
            str(args.ev_bridge_bundle)
            if args.ev_bridge_bundle is not None
            else (
                immediate_calibration["immediate_calibration_ev_bridge_bundle"]
                if immediate_calibration is not None else None
            )
        ),
        "ev_bridge_bundle_sha256": (
            str(bridge_bundle.manifest.get("bundle_sha256"))
            if bridge_bundle is not None else None
        ),
        "immediate_calibration": immediate_calibration,
        "ev_mapping_score_family_id": str(
            admitted["ev_mapping_score_family_id"].iloc[0]
        ),
        "ev_mapping_conversion_vintage": str(
            admitted["ev_mapping_conversion_vintage"].iloc[0]
        ),
        "ev_mapping_upstream_vintage": str(
            admitted["ev_mapping_upstream_vintage"].iloc[0]
        ),
        "resolved_producer_lineage": (
            str(args.resolved_producer_lineage)
            if args.resolved_producer_lineage is not None else "embedded_in_ledger"
        ),
        "rows": int(len(admitted)),
        "decision_ts": decision_ts.isoformat(),
        "resolved_rows_available_before_decision": int(len(resolved)),
        "resolved_rows_compatible_with_active_score_vintage": compatible_resolved_rows,
        "mapped_rows": int(admitted["causal_21d_side_expected_net_bps"].notna().sum()),
        "admitted_rows": int(admitted["causal_21d_side_admitted_ge_50bps"].fillna(False).sum()),
        "feature_complete_current_rows": int((~incomplete).sum()),
        "feature_incomplete_current_rows": int(incomplete.sum()),
        "geometry_bundle_sha256": next(iter(current_geometry)),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
