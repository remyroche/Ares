#!/usr/bin/env python3
"""Audit one schema-v6 live hour against its immutable predecessor."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_RELATIVE = 0.0001  # 0.01%
OUTPUT_FIELDS = (
    "base_score", "base_rank42", "base_anchor_bps",
    "k9_entropy", "k9_top2_margin", "k9_ood_distance",
    "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
    "upstream", "correctness_raw", "correctness_rank", "final_score",
    "robust21_expected_net_bps", "mc1_d2_expected_net_bps",
    "mc1_d2_recent_global_shift_bps",
    "a5_bounded10_expected_bps", "a5_calibrated_expected_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _recent_performance(ledger: pd.DataFrame, decision: pd.Timestamp) -> dict[str, object]:
    work = ledger.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True)
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True,
    )
    work = work[
        work["policy_label_available_ts"].le(decision)
        & work["policy_path_valid"].fillna(False).astype(bool)
    ].copy()
    work["final_score"] = pd.to_numeric(work["final_score"], errors="coerce")
    work["base_score"] = pd.to_numeric(work["base_score"], errors="coerce")
    work["policy_net_bps"] = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    work = work.dropna(subset=["final_score", "base_score", "policy_net_bps"])
    results: dict[str, object] = {}
    for days in (3, 7, 14, 28):
        recent = work[work["__decision_ts__"].ge(decision - pd.Timedelta(days=days))].copy()
        if recent.empty:
            results[f"{days}d"] = {"rows": 0}
            continue
        within_ts = recent.groupby("__decision_ts__")["final_score"].rank(
            pct=True, method="first",
        )
        tail = recent[within_ts.ge(0.80)]
        results[f"{days}d"] = {
            "rows": int(len(recent)),
            "final_score_rank_ic": float(recent["final_score"].corr(recent["policy_net_bps"], method="spearman")),
            "base_score_rank_ic": float(recent["base_score"].corr(recent["policy_net_bps"], method="spearman")),
            "timestamp_top20_rows": int(len(tail)),
            "timestamp_top20_hit_rate": float(tail["policy_net_bps"].gt(0).mean()),
            "timestamp_top20_net_bps": float(tail["policy_net_bps"].mean()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--enforce-live-wall-clock", action="store_true")
    parser.add_argument(
        "--late-entry-execution-contract", type=Path, default=None,
        help=(
            "Hash-bound execution contract for one explicit late-entry "
            "exception. Its decision timestamp and expiry are verified here."
        ),
    )
    parser.add_argument(
        "--maximum-audit-age-seconds", type=float, default=None,
        help=(
            "Post-decision audit deadline. This may exceed the immutable entry "
            "deadline because the audit is read-only and cannot authorize trades."
        ),
    )
    args = parser.parse_args()
    audit_started_at = pd.Timestamp(datetime.now(timezone.utc))
    if args.out.exists():
        raise FileExistsError(f"immutable live-hour audit exists: {args.out}")
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    cycle = json.loads((args.run / "cycle/run_manifest.json").read_text())
    previous_manifest = json.loads((args.previous_run / "run_manifest.json").read_text())
    previous_cycle = json.loads((args.previous_run / "cycle/run_manifest.json").read_text())
    decision = _utc(manifest["decision_ts"])
    previous_decision = _utc(previous_manifest["decision_ts"])
    late_override = False
    late_contract_sha256: str | None = None
    if args.late_entry_execution_contract is not None:
        payload = json.loads(args.late_entry_execution_contract.read_text())
        declared_decision = payload.get("late_entry_override_decision_ts")
        declared_expiry = payload.get("late_entry_override_expires_at")
        if (
            not bool(payload.get("order_submission_authorized"))
            or declared_decision is None
            or declared_expiry is None
            or _utc(declared_decision) != decision
            or pd.Timestamp(datetime.now(timezone.utc)) > _utc(declared_expiry)
            or str(payload.get("inference_bundle_sha256"))
            != str(manifest["hashes"]["inference_bundle"])
        ):
            raise ValueError("late-entry execution contract does not authorize this receipt")
        late_override = True
        late_contract_sha256 = _sha(args.late_entry_execution_contract)
    completed_inside_window = (
        bool(manifest.get("live_wall_clock_enforced"))
        and bool(manifest.get("completed_within_live_decision_window"))
        and 0.0 <= float(manifest.get("decision_age_at_start_seconds", np.inf))
        <= float(manifest.get("live_decision_freshness_seconds", -1))
        and 0.0 <= float(manifest.get("decision_age_at_completion_seconds", np.inf))
        <= float(manifest.get("live_decision_freshness_seconds", -1))
    )
    checks = {
        "completed_inside_live_decision_window": completed_inside_window or late_override,
        "schema_v6_bundle": (
            manifest["inference_bundle_audit"]["schema"]
            == "strict_r3_inference_bundle_v6_robust21_mc1_d2_adaptive_exit_v1"
        ),
        "artifact_hashes_verified": int(
            manifest["inference_bundle_audit"]["hashes_verified"]
        ) == 28,
        "runtime_hashes_verified": int(
            manifest["inference_bundle_audit"]["runtime_code_hashes_verified"]
        ) >= 39,
        "future_paths_absent": manifest.get("future_paths_consumed") == [],
        "exchange_free_receipt": (
            int(manifest.get("exchange_calls", -1)) == 0
            and not bool(manifest.get("order_submission_enabled"))
        ),
        "all_cycle_checks": all(bool(value) for value in cycle["checks"].values()),
        "current_feature_parity": all(
            bool(value)
            for value in manifest["current_feature_parity_audit"]["checks"].values()
        ),
        "row_local_missing_data": (
            manifest["row_local_feature_skip_audit"]["reason"]
            == "feature_unavailable_at_decision"
        ),
        "frozen_geometry": (
            manifest["inference_bundle_audit"]["geometry_bundle_sha256"]
            == "dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638"
        ),
        "persisted_feature_state_only": (
            manifest["inference_bundle_audit"].get(
                "persisted_feature_state_mode"
            ) == "persisted_state_only"
            and bool(manifest.get("persisted_feature_state_only"))
            and bool(manifest.get("stateful_feature_bundle_input"))
            and bool(manifest.get("stateful_feature_bundle_output"))
            and bool(manifest["inference_bundle_audit"].get(
                "final14_contract_sha256"
            ))
        ),
        "policy_cost_once": bool(cycle["checks"].get("policy_cost_once")),
        "prior_resolved_calibration": (
            bool(cycle["checks"].get("runtime_labels_strictly_prior_day"))
            and bool(cycle["checks"].get("current_decision_labels_excluded"))
        ),
        "daily_calibration_update_cadence": (
            cycle["runtime_resolved_state_sha256"]
            != previous_cycle["runtime_resolved_state_sha256"]
        ) == (decision.normalize() != previous_decision.normalize()),
    }
    overlap = manifest["append_only_overlap_audit"]
    checks["exact_append_only_inputs_outputs"] = all(
        item.get("changed_fields") == []
        and float(item.get("max_numeric_delta", np.inf)) == 0.0
        for item in overlap.values()
    )

    decisions = pd.read_parquet(args.run / "cycle/shadow_decisions.parquet")
    current = decisions.loc[
        pd.to_datetime(decisions["__decision_ts__"], utc=True).eq(decision)
    ].copy()
    entry_data_state = str(manifest.get("current_entry_data_state") or "")
    checks["current_entry_data_state"] = (
        (
            entry_data_state == "actionable"
            and int(manifest.get("current_feature_parity_rows", 0)) > 0
            and not current.empty
        )
        or (
            entry_data_state == "no_actionable_rows_fail_closed"
            and int(manifest.get("current_feature_parity_rows", -1)) == 0
            and current.empty
            and int(manifest.get("mapped_rows", -1)) == 0
            and int(manifest.get("admitted_rows", -1)) == 0
            and int(manifest.get("portfolio_accepted_rows", -1)) == 0
            and bool(
                manifest["current_feature_parity_audit"]["checks"].get(
                    "empty_current_entry_set_failed_closed"
                )
            )
        )
    )
    route = current["base_route_timestamp"].fillna(False).astype(bool)
    route_top30 = current["base_route_timestamp_top30"].fillna(False).astype(bool)
    route_legacy = current["base_route_timestamp_top20"].fillna(False).astype(bool)
    route_fraction = pd.to_numeric(current["base_route_fraction"], errors="coerce")
    admitted = current["mc1_d2_admitted_ge_50bps"].fillna(False).astype(bool)
    accepted = current["portfolio_accepted"].fillna(False).astype(bool)
    checks.update({
        "configured_top30_route_contract": (
            route.eq(route_top30).all()
            and route.eq(route_legacy).all()
            and route_fraction.eq(0.30).all()
            and (~admitted | route).all()
            and (~accepted | route).all()
        ),
        "mc1_admission_contract": (~accepted | admitted).all(),
        "a5_outputs_present_for_routed": (
            {"a5_bounded10_expected_bps", "a5_bounded10_available"}
            .issubset(current.columns)
            and pd.to_numeric(
                current.loc[route, "a5_bounded10_expected_bps"],
                errors="coerce",
            ).notna().all()
            and current.loc[route, "a5_bounded10_available"].fillna(False).all()
        ),
        "long_only": current["side_name"].astype(str).str.lower().eq("long").all(),
        "same_policy": (
            current["policy_sl_atr"].eq(4.15200064332387).all()
            and current["policy_trailing_activation_atr"].eq(2.326224919759605).all()
            and current["policy_trailing_giveback_atr"].eq(0.10237198997143725).all()
            and current["policy_timeout_hours"].eq(12).all()
            and current["policy_cost_bps_once"].eq(100.0).all()
        ),
    })
    if not all(checks.values()):
        raise AssertionError(f"schema-v6 live-hour audit failed: {checks}")
    telemetry = {
        field: {
            "finite_rows": int(pd.to_numeric(current[field], errors="coerce").notna().sum()),
            "minimum": float(pd.to_numeric(current[field], errors="coerce").min()),
            "maximum": float(pd.to_numeric(current[field], errors="coerce").max()),
        }
        for field in OUTPUT_FIELDS if field in current
    }
    previous_decisions = pd.read_parquet(args.previous_run / "cycle/shadow_decisions.parquet")
    previous_current = previous_decisions.loc[
        pd.to_datetime(previous_decisions["__decision_ts__"], utc=True).eq(previous_decision)
    ]
    previous_shift = pd.to_numeric(
        previous_current.get("mc1_d2_recent_global_shift_bps"), errors="coerce",
    ).dropna()
    current_shift = pd.to_numeric(
        current.get("mc1_d2_recent_global_shift_bps"), errors="coerce",
    ).dropna()
    ledger = pd.read_parquet(
        args.run / "cycle/runtime_resolved_state/walkforward_scored_label_ledger.parquet",
        columns=[
            "__decision_ts__", "policy_label_available_ts", "policy_path_valid",
            "final_score", "base_score", "policy_net_bps",
        ],
    )
    resolved_manifest = json.loads(
        (args.run / "cycle/runtime_resolved_state/run_manifest.json").read_text()
    )
    result = {
        "schema": "strict_r3_schema_v6_live_hour_audit_v1",
        "status": "pass",
        "decision_ts": decision.isoformat(),
        "run": str(args.run), "previous_run": str(args.previous_run),
        "checks": checks, "tolerance_relative": TOLERANCE_RELATIVE,
        "late_entry_override": {
            "used": late_override,
            "normal_freshness_passed": completed_inside_window,
            "execution_contract_sha256": late_contract_sha256,
        },
        "rows": int(len(current)), "routed_rows": int(route.sum()),
        "current_entry_data_state": entry_data_state,
        "mc1_admitted_rows": int(admitted.sum()),
        "portfolio_accepted_rows": int(accepted.sum()),
        "missing_data_skipped_rows": int(
            manifest["row_local_feature_skip_audit"]["skipped_rows"]
        ),
        "maximum_historical_numeric_delta": max(
            float(item["max_numeric_delta"]) for item in overlap.values()
        ),
        "calibration": {
            "runtime_resolved_state_sha256": cycle["runtime_resolved_state_sha256"],
            "previous_runtime_resolved_state_sha256": previous_cycle["runtime_resolved_state_sha256"],
            "runtime_state_updated": cycle["runtime_resolved_state_sha256"] != previous_cycle["runtime_resolved_state_sha256"],
            "newly_appended_resolved_rows": int(resolved_manifest["newly_appended_rows"]),
            "maximum_label_available_ts": resolved_manifest["max_label_available_ts"],
            "robust21_control_admitted_rows": cycle["robust21_control_admitted_rows"],
            "mc1_recent_global_shift_present_rows": int(
                pd.to_numeric(current["mc1_d2_recent_global_shift_bps"], errors="coerce")
                .notna().sum()
            ),
            "mc1_recent_global_shift_bps": (
                float(current_shift.iloc[0]) if len(current_shift) else None
            ),
            "previous_mc1_recent_global_shift_bps": (
                float(previous_shift.iloc[0]) if len(previous_shift) else None
            ),
            "recent_realized_performance": _recent_performance(ledger, decision),
        },
        "output_telemetry": telemetry,
        "run_manifest_sha256": _sha(args.run / "run_manifest.json"),
    }
    audit_completed_at = pd.Timestamp(datetime.now(timezone.utc))
    inference_freshness_seconds = float(
        manifest["live_decision_freshness_seconds"]
    )
    audit_maximum_age_seconds = (
        inference_freshness_seconds
        if args.maximum_audit_age_seconds is None
        else float(args.maximum_audit_age_seconds)
    )
    if audit_maximum_age_seconds < inference_freshness_seconds:
        raise ValueError(
            "maximum audit age cannot be shorter than the frozen inference window"
        )
    audit_completion_age = float((audit_completed_at - decision).total_seconds())
    audit_within_window = (
        0.0 <= audit_completion_age <= audit_maximum_age_seconds
    )
    result.update({
        "audit_started_at": audit_started_at.isoformat(),
        "audit_completed_at": audit_completed_at.isoformat(),
        "audit_completion_decision_age_seconds": audit_completion_age,
        "inference_freshness_seconds": inference_freshness_seconds,
        "audit_maximum_age_seconds": audit_maximum_age_seconds,
        "audit_completed_within_audit_window": audit_within_window,
        # Kept for compatibility with older receipts; this is the read-only
        # audit window, not the entry/scoring freshness contract.
        "audit_completed_within_live_decision_window": audit_within_window,
        "live_wall_clock_enforced": bool(args.enforce_live_wall_clock),
    })
    if args.enforce_live_wall_clock and not audit_within_window:
        result["status"] = "fail_late_audit"
    args.out.mkdir(parents=True)
    (args.out / "run_manifest.json").write_text(
        json.dumps(result, indent=2, default=lambda value: value.item()) + "\n"
    )
    current.to_parquet(args.out / "current_decisions.parquet", index=False)
    print(json.dumps(result, default=lambda value: value.item()))
    if result["status"] != "pass":
        raise SystemExit("live-hour audit completed outside decision window")


if __name__ == "__main__":
    main()
