#!/usr/bin/env python3
"""Rebind a non-flat strict-R3 live ledger to a reviewed runtime successor.

This migration is narrower than a model migration. Frozen artifacts, feature
contract, model/calibration/geometry inputs, admission, policy, economics and
portfolio semantics must remain identical. The only permitted inference
change is runtime implementation/state lineage, including the reviewed causal
no-trade 15-minute representation. Actual positions and processed decisions
are copied byte-for-byte as JSON values into a new immutable state file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIGRATION_KIND = "causal_no_trade_15m_runtime_successor_v1"
BOUNDED_FRESHNESS_EXTENSION_KIND = "bounded_current_hour_freshness_extension_v1"
APPROVED_MODEL_PROMOTION_KIND = "approved_dual_admission_model_promotion_v1"
APPROVED_POLICY_PROMOTION_KIND = "approved_smooth_capital_protection_policy_v1"
APPROVED_EXECUTION_FRICTION_REVISION_KIND = "approved_live_microstructure_buffer_revision_v1"
APPROVED_EXIT_VWAP_STOP_PROTECTION_KIND = (
    "approved_exit_vwap_protective_stop_revision_v1"
)
APPROVED_DUAL_ADMISSION_ONLY_EXECUTION_AUTHORITY_KIND = (
    "approved_dual_admission_only_execution_authority_v1"
)
APPROVED_EXECUTION_ADJUSTED_EV_30BPS_GATE_KIND = (
    "approved_execution_adjusted_ev_30bps_gate_v1"
)
APPROVED_CLOSE_BASED_HARD_STOP_MONITOR_KIND = (
    "approved_completed_minute_close_based_hard_stop_monitor_v1"
)
APPROVED_EXECUTABLE_VWAP_FROZEN_THRESHOLD_SENTINEL_KIND = (
    "approved_executable_vwap_frozen_threshold_exit_sentinel_v1"
)
APPROVED_OPENPOSITIONS_503_FILLS_FALLBACK_KIND = (
    "approved_openpositions_503_fills_position_monitor_fallback_v1"
)
APPROVED_DIRECT_15M_BOOK_GUARD_KIND = (
    "approved_direct_15m_decision_open_book_guard_v1"
)
APPROVED_OI_FUNDING_SIDECAR_CONTAINMENT_KIND = (
    "approved_oi_funding_corrupt_sidecar_containment_v1"
)
DIRECT_15M_BOOK_GUARD_RUNTIME_PATH = (
    "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py"
)
OI_FUNDING_SIDECAR_CONTAINMENT_RUNTIME_PATHS = {
    "scripts/backfill_kraken_oi_funding_sidecars.py",
    "scripts/run_strict_r3_live_hourly_entry_producer.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Write the migration outputs atomically without importing live I/O code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_successor_live_authorization(execution: dict) -> None:
    """Validate the sealed live authority structurally for a state rebind.

    State migration must not initialise CCXT or a data-store client.  The live
    producer/monitor still load the full execution contract before any venue
    call.  Here we bind the exact authorization file, its hash, the successor
    inference hash and the fresh-signal prohibition needed for a safe ledger
    successor.
    """
    relative = str(execution.get("activation_authorization") or "")
    expected = str(execution.get("activation_authorization_sha256") or "")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file() or _sha(path) != expected:
        raise ValueError("successor activation authorization is missing or hash-mismatched")
    payload = json.loads(path.read_text())
    if (
        payload.get("authorized") is not True
        or payload.get("stale_signal_execution_prohibited") is not True
        or str(payload.get("inference_bundle_sha256") or "")
        != str(execution.get("inference_bundle_sha256") or "")
    ):
        raise ValueError("successor activation authorization does not bind the live contract")


def _resolve_execution_payload(path: Path) -> dict:
    """Resolve the narrow runtime-only execution successor overlay."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_kraken_live_execution_overlay_v1":
        return payload
    base_rel = payload.get("base_execution")
    if not isinstance(base_rel, str) or not base_rel:
        raise ValueError("execution overlay lacks base execution")
    base_path = (ROOT / base_rel).resolve()
    if ROOT.resolve() not in base_path.parents or not base_path.is_file():
        raise ValueError("execution overlay base escapes repository root")
    # A successor may itself be a narrow execution overlay.  Resolve it
    # recursively so a state migration follows the same immutable semantic
    # chain as the live loader rather than rejecting an otherwise-valid
    # runtime-only reseal.
    base = _resolve_execution_payload(base_path)
    if base.get("schema") != "strict_r3_kraken_live_execution_v1":
        raise ValueError("execution overlay has an unexpected base schema")
    overrides = payload.get("overrides") or {}
    allowed = {
        "inference_bundle", "inference_bundle_sha256",
        "activation_authorization", "activation_authorization_sha256",
        "runtime_code_sha256",
    }
    if not isinstance(overrides, dict) or set(overrides).difference(allowed):
        raise ValueError("execution overlay has unsupported overrides")
    merged = dict(base)
    for key, value in overrides.items():
        if key == "runtime_code_sha256":
            hashes = dict(base.get(key) or {})
            hashes.update(dict(value or {}))
            merged[key] = hashes
        else:
            merged[key] = value
    return merged


def _resolve_overlay_semantics_without_runtime_load(path: Path) -> dict:
    """Resolve a prior overlay structurally when its code hashes are stale.

    This is migration-only and intentionally validates the base/allow-list
    merge rather than executing or accepting the old runtime implementation.
    It exists so a reseal of reporting/monitor code cannot make a non-flat
    ledger impossible to migrate to its hash-bound successor.
    """
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_inference_bundle_overlay_v1":
        return payload
    base_rel = payload.get("base_bundle")
    if not isinstance(base_rel, str) or not base_rel:
        raise ValueError("prior overlay lacks base bundle")
    base_path = (ROOT / base_rel).resolve()
    if ROOT.resolve() not in base_path.parents or not base_path.is_file():
        raise ValueError("prior overlay base escapes repository root")
    base = json.loads(base_path.read_text())
    if str(base.get("schema")) != "strict_r3_inference_bundle_v6_robust21_mc1_d2_adaptive_exit_v1":
        raise ValueError("prior overlay has an unexpected base schema")
    overrides = payload.get("overrides") or {}
    allowed = {
        "admission_contract", "runtime", "paths", "sha256",
        "runtime_code_sha256", "dual_bcf_current",
    }
    if not isinstance(overrides, dict) or set(overrides).difference(allowed):
        raise ValueError("prior overlay has unsupported overrides")
    merged = dict(base)
    for key, value in overrides.items():
        if key in {"runtime", "paths", "sha256", "runtime_code_sha256"}:
            combined = dict(base.get(key) or {})
            combined.update(dict(value or {}))
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def _execution_economics(payload: dict) -> dict:
    excluded = {
        "inference_bundle", "inference_bundle_sha256",
        "activation_authorization", "activation_authorization_sha256",
        "runtime_code_sha256", "live_shadow_bridge_contract",
        "exit_replay_contract", "runtime_checkpoint_required_before_order_submission",
        # Presentation-only close-email lineage is hash-bound and audited, but
        # it cannot alter a decision, order, position, policy, or economics.
        "close_reporting",
        # These fields document reviewed runtime lineage.  They neither alter
        # execution economics nor relax a policy/admission/portfolio gate, and
        # must therefore be allowed to advance with a sealed runtime successor.
        "runtime_reseal_predecessors", "version_note",
        # Human-readable lineage strings are not consumed by the executor.
        # Keep state migration focused on executable parameters/hashes.
        "execution_adjusted_ev", "position_monitor", "protective_stop",
        # A validated recovery predecessor constrains the producer's state
        # lineage, not live entry/exit economics.  It may be added when a
        # flat state is moved onto a completed recovery successor.
        "stateful_recovery_successor",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _inference_semantics(payload: dict) -> dict:
    runtime = dict(payload.get("runtime") or {})
    runtime_fields = (
        "adaptive_exit", "admission", "base_route",
        "candidate_feature_population", "candidate_materializer",
        "current_spread_gate", "entry_open_contract", "entry_price_lineage",
        "feature_edge_contract", "feature_history_start", "late_source_policy",
        "oi_refresh_contract", "policy_bar_root", "resolved_calibration_update",
        "score_chunk_hours", "shadow_cycle",
    )
    return {
        "schema": payload.get("schema"),
        "scope": payload.get("scope"),
        "activation_ts": payload.get("activation_ts"),
        "end_exclusive_ts": payload.get("end_exclusive_ts"),
        "outside_window": payload.get("outside_window"),
        "live_decision_freshness_seconds": payload.get(
            "live_decision_freshness_seconds"
        ),
        "missing_entry_data_contract": payload.get("missing_entry_data_contract"),
        "reference_window_days": payload.get("reference_window_days"),
        "ev_bridge_role": payload.get("ev_bridge_role"),
        "admission_contract": payload.get("admission_contract"),
        "trust_overlay_contract": payload.get("trust_overlay_contract"),
        "resolved_outcome_contract": payload.get("resolved_outcome_contract"),
        "adaptive_exit_contract": payload.get("adaptive_exit_contract"),
        "adaptive_exit_role": payload.get("adaptive_exit_role"),
        "paths": payload.get("paths"),
        "sha256": payload.get("sha256"),
        "producer": payload.get("producer"),
        "feature_parity": payload.get("feature_parity"),
        "runtime_semantics": {key: runtime.get(key) for key in runtime_fields},
    }


def _only_direct_15m_book_guard(
    *, old_inference: dict, new_inference: dict,
) -> bool:
    """Allow exactly the reviewed target-free stale-price eligibility repair."""
    if _inference_semantics(old_inference) != _inference_semantics(new_inference):
        return False
    old_hashes = dict(old_inference.get("runtime_code_sha256") or {})
    new_hashes = dict(new_inference.get("runtime_code_sha256") or {})
    changed = {
        key for key in set(old_hashes).union(new_hashes)
        if old_hashes.get(key) != new_hashes.get(key)
    }
    return changed == {DIRECT_15M_BOOK_GUARD_RUNTIME_PATH}


def _only_oi_funding_sidecar_containment(
    *, old_inference: dict, new_inference: dict,
) -> bool:
    """Allow only a local corrupt-source fail-closed containment repair."""
    if _inference_semantics(old_inference) != _inference_semantics(new_inference):
        return False
    old_hashes = dict(old_inference.get("runtime_code_sha256") or {})
    new_hashes = dict(new_inference.get("runtime_code_sha256") or {})
    changed = {
        key for key in set(old_hashes).union(new_hashes)
        if old_hashes.get(key) != new_hashes.get(key)
    }
    return changed == OI_FUNDING_SIDECAR_CONTAINMENT_RUNTIME_PATHS


COMPACT_EXECUTION_1M_RANGE_PRUNING_RUNTIME_PATH = (
    "extreme_price_movements/data_store.py"
)


def _only_compact_execution_1m_range_pruning(
    *, old_inference: dict, new_inference: dict,
) -> bool:
    """Allow only bounded pruning of an unambiguous legacy compact part.

    The repair changes no source values or execution thresholds. It merely
    prevents a current bounded read from decoding a compacted historical year
    whose filename already proves it cannot overlap the requested interval.
    """
    if _inference_semantics(old_inference) != _inference_semantics(new_inference):
        return False
    old_hashes = dict(old_inference.get("runtime_code_sha256") or {})
    new_hashes = dict(new_inference.get("runtime_code_sha256") or {})
    changed = {
        key for key in set(old_hashes).union(new_hashes)
        if old_hashes.get(key) != new_hashes.get(key)
    }
    return changed == {COMPACT_EXECUTION_1M_RANGE_PRUNING_RUNTIME_PATH}


def _only_bounded_current_hour_extension(
    *, old_execution: dict, new_execution: dict,
    old_inference: dict, new_inference: dict,
) -> bool:
    """Allow exactly the reviewed 15-minute → current-hour freshness extension.

    This is deliberately narrower than an ordinary runtime successor.  It
    permits no model, feature, policy, admission, economics, or portfolio
    change: only the live-wall-clock ceiling and its matching explanatory
    runtime text may differ.  It is intended for a restored live service that
    must finish a valid current-hour decision after its initial 15-minute
    window, never to resubmit an earlier-hour signal.
    """
    if (
        int(old_execution.get("maximum_decision_age_seconds", -1)) != 900
        or int(new_execution.get("maximum_decision_age_seconds", -1)) != 3600
        or int(old_inference.get("live_decision_freshness_seconds", -1)) != 900
        or int(new_inference.get("live_decision_freshness_seconds", -1)) != 3600
    ):
        return False
    old_semantics = _inference_semantics(old_inference)
    new_semantics = _inference_semantics(new_inference)
    old_semantics["live_decision_freshness_seconds"] = "__bounded_extension__"
    new_semantics["live_decision_freshness_seconds"] = "__bounded_extension__"
    old_runtime = old_semantics["runtime_semantics"]
    new_runtime = new_semantics["runtime_semantics"]
    old_runtime["late_source_policy"] = "__bounded_extension__"
    new_runtime["late_source_policy"] = "__bounded_extension__"
    return old_semantics == new_semantics


def migrate_runtime_successor_state(
    *,
    state: dict,
    old_execution: dict,
    new_execution: dict,
    old_inference: dict,
    new_inference: dict,
    allow_bounded_current_hour_extension: bool = False,
    allow_approved_model_promotion: bool = False,
    allow_approved_policy_promotion: bool = False,
    allow_approved_execution_friction_revision: bool = False,
    allow_approved_exit_vwap_stop_protection: bool = False,
    allow_approved_close_based_hard_stop_monitor: bool = False,
    allow_approved_executable_vwap_frozen_threshold_sentinel: bool = False,
    allow_approved_dual_admission_only_execution_authority: bool = False,
    allow_approved_execution_adjusted_ev_30bps_gate: bool = False,
    allow_approved_openpositions_503_fills_fallback: bool = False,
    allow_approved_direct_15m_book_guard: bool = False,
    allow_approved_oi_funding_sidecar_containment: bool = False,
    allow_approved_compact_execution_1m_range_pruning: bool = False,
) -> dict:
    expected_old = {
        "inference_bundle_sha256": old_execution["inference_bundle_sha256"],
        "exit_policy_sha256": old_execution["exit_policy_sha256"],
        "activation_authorization_sha256": old_execution[
            "activation_authorization_sha256"
        ],
    }
    for field, expected in expected_old.items():
        if state.get(field) != expected:
            raise ValueError(f"source live state does not match old contract: {field}")
    if bool(old_execution.get(
        "runtime_checkpoint_required_before_order_submission", False
    )) and not bool(new_execution.get(
        "runtime_checkpoint_required_before_order_submission", False
    )):
        raise ValueError("successor may not weaken the runtime-checkpoint gate")
    if not bool(new_execution.get(
        "runtime_checkpoint_required_before_order_submission", False
    )):
        raise ValueError("successor requires a pre-order runtime checkpoint")
    old_economics = _execution_economics(old_execution)
    new_economics = _execution_economics(new_execution)
    bounded_extension = _only_bounded_current_hour_extension(
        old_execution=old_execution,
        new_execution=new_execution,
        old_inference=old_inference,
        new_inference=new_inference,
    )
    if allow_bounded_current_hour_extension and not bounded_extension:
        raise ValueError(
            "requested freshness extension is not exactly 900→3600 seconds "
            "with otherwise identical inference semantics"
        )
    if bounded_extension and not allow_bounded_current_hour_extension:
        raise ValueError(
            "bounded current-hour freshness extension requires explicit CLI approval"
        )
    if bounded_extension:
        old_economics.pop("maximum_decision_age_seconds", None)
        new_economics.pop("maximum_decision_age_seconds", None)
    policy_promotion = (
        allow_approved_policy_promotion
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and str(new_execution.get("exit_policy_sha256"))
        != str(old_execution.get("exit_policy_sha256"))
    )
    if policy_promotion:
        # The explicit user-approved successor changes only the hash-bound
        # parent exit policy and its policy-monitor metadata.  Every model,
        # admission, portfolio and entry-economics field stays identical.
        old_economics.pop("exit_policy", None)
        old_economics.pop("exit_policy_sha256", None)
        new_economics.pop("exit_policy", None)
        new_economics.pop("exit_policy_sha256", None)
        old_economics.pop("version_note", None)
        new_economics.pop("version_note", None)
    friction_revision = (
        allow_approved_execution_friction_revision
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and "max(80 bps" in str(old_execution.get("execution_adjusted_ev") or "")
        and "microstructure friction + adverse delay gap + 10 bps"
        in str(new_execution.get("execution_adjusted_ev") or "")
        and float(new_execution.get("execution_microstructure_buffer_bps", -1.0)) == 10.0
    )
    if friction_revision:
        for payload in (old_economics, new_economics):
            payload.pop("execution_adjusted_ev", None)
            payload.pop("execution_microstructure_buffer_bps", None)
            payload.pop("version_note", None)
    exit_vwap_stop_revision = (
        allow_approved_exit_vwap_stop_protection
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and bool(new_execution.get("protective_stop_exit_vwap_adjustment", False))
        and not bool(old_execution.get("protective_stop_exit_vwap_adjustment", False))
        and int(new_execution.get("protective_stop_book_levels", 10)) >= 1
    )
    if exit_vwap_stop_revision:
        # The user-approved change raises the native trigger by the observable
        # full-size sell-side VWAP impact so that the policy stop is an
        # intended executable price.  Model/admission/portfolio semantics and
        # the rich policy target remain unchanged.
        for payload in (old_economics, new_economics):
            payload.pop("protective_stop_exit_vwap_adjustment", None)
            payload.pop("protective_stop_book_levels", None)
            payload.pop("protective_stop", None)
            payload.pop("version_note", None)
    admission_only_execution_authority = (
        allow_approved_dual_admission_only_execution_authority
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and not bool(old_execution.get("execution_book_telemetry_only", False))
        and bool(new_execution.get("execution_book_telemetry_only", False))
        and str(new_execution.get("execution_entry_authority"))
        == "sealed_dual_mc1_admission_then_common_portfolio_auction_only"
    )
    if admission_only_execution_authority:
        # Explicit user-approved change: causal dual-MC1 admission plus the
        # common portfolio auction become the complete entry authority.  The
        # live book remains mandatory for order sizing/protection and durable
        # telemetry, but cannot veto or rerank an auction winner.
        for payload in (old_economics, new_economics):
            payload.pop("execution_book_telemetry_only", None)
            payload.pop("execution_entry_authority", None)
            payload.pop("version_note", None)
    execution_adjusted_ev_30bps_gate = (
        allow_approved_execution_adjusted_ev_30bps_gate
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and bool(old_execution.get("execution_book_telemetry_only", False))
        and bool(new_execution.get("execution_book_telemetry_only", False))
        and not bool(old_execution.get("execution_adjusted_ev_veto_enabled", False))
        and bool(new_execution.get("execution_adjusted_ev_veto_enabled", False))
        and float(old_execution.get("minimum_execution_adjusted_ev_bps", 50.0)) == 50.0
        and float(new_execution.get("minimum_execution_adjusted_ev_bps", 50.0)) == 30.0
        and str(new_execution.get("execution_entry_authority"))
        == "sealed_dual_mc1_admission_then_common_portfolio_auction_then_execution_adjusted_ev_ge_30bps"
    )
    if execution_adjusted_ev_30bps_gate:
        # Explicitly approved safety restoration: candidate selection remains
        # dual MC1 admission plus the common portfolio auction.  Only the
        # already-selected winner is vetoed when live execution economics no
        # longer clear +30 bps; it is never reranked by the order book.
        for payload in (old_economics, new_economics):
            payload.pop("execution_adjusted_ev_veto_enabled", None)
            payload.pop("minimum_execution_adjusted_ev_bps", None)
            payload.pop("execution_entry_authority", None)
            payload.pop("version_note", None)
    close_based_hard_stop_monitor = (
        allow_approved_close_based_hard_stop_monitor
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and not bool(old_execution.get("full_vwap_hard_stop_monitor_enabled", False))
        and not bool(new_execution.get("full_vwap_hard_stop_monitor_enabled", False))
        and not bool(old_execution.get("close_based_hard_stop_monitor_enabled", False))
        and bool(new_execution.get("close_based_hard_stop_monitor_enabled", False))
    )
    if close_based_hard_stop_monitor:
        # User approved returning to the prior completed-1m close controller
        # for the initial hard stop.  The full-VWAP controller remains disabled
        # and every model, admission, auction, entry, native-stop and parent
        # rich-policy parameter is unchanged.
        for payload in (old_economics, new_economics):
            payload.pop("full_vwap_hard_stop_monitor_enabled", None)
            payload.pop("close_based_hard_stop_monitor_enabled", None)
            payload.pop("position_monitor", None)
            payload.pop("version_note", None)
    executable_vwap_frozen_threshold_sentinel = (
        allow_approved_executable_vwap_frozen_threshold_sentinel
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and not bool(old_execution.get(
            "executable_vwap_frozen_threshold_sentinel_enabled", False
        ))
        and bool(new_execution.get(
            "executable_vwap_frozen_threshold_sentinel_enabled", False
        ))
        and not bool(new_execution.get("full_vwap_hard_stop_monitor_enabled", False))
        and bool(new_execution.get("close_based_hard_stop_monitor_enabled", False))
        and bool(new_execution.get("protective_stop_exit_vwap_adjustment", False))
        and float(new_execution.get("native_last_stop_backstop_bps", -1.0))
        == float(new_execution.get("maximum_exit_slippage_bps", -2.0))
    )
    if executable_vwap_frozen_threshold_sentinel:
        # Explicitly approved live-only exit safeguard.  It has no model,
        # feature, score, admission, auction, entry, or parent-policy target
        # authority; it only preempts the resident lower native stop with a
        # fresh remaining-size executable VWAP against the prior frozen bar
        # threshold.
        for payload in (old_economics, new_economics):
            payload.pop("executable_vwap_frozen_threshold_sentinel_enabled", None)
            payload.pop("native_last_stop_backstop_bps", None)
            payload.pop("position_monitor", None)
            payload.pop("version_note", None)
    openpositions_503_fills_fallback = (
        allow_approved_openpositions_503_fills_fallback
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _inference_semantics(old_inference) == _inference_semantics(new_inference)
        and not bool(old_execution.get(
            "openpositions_503_fills_fallback_enabled", False
        ))
        and bool(new_execution.get(
            "openpositions_503_fills_fallback_enabled", False
        ))
    )
    if openpositions_503_fills_fallback:
        # User-approved availability repair: the minute monitor may recover
        # only persisted tracked inventory from the authenticated /fills
        # ledger when, and only when, Kraken's /openpositions returns 503.
        # It cannot discover positions, open entries, relax exits, or change
        # model, admission, portfolio, or parent-policy semantics.
        for payload in (old_economics, new_economics):
            payload.pop("openpositions_503_fills_fallback_enabled", None)
            payload.pop("position_monitor", None)
            payload.pop("version_note", None)
    direct_15m_book_guard = (
        allow_approved_direct_15m_book_guard
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _only_direct_15m_book_guard(
            old_inference=old_inference, new_inference=new_inference,
        )
    )
    oi_funding_sidecar_containment = (
        allow_approved_oi_funding_sidecar_containment
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _only_oi_funding_sidecar_containment(
            old_inference=old_inference, new_inference=new_inference,
        )
    )
    compact_execution_1m_range_pruning = (
        allow_approved_compact_execution_1m_range_pruning
        and str(old_execution.get("exit_policy_sha256"))
        == str(new_execution.get("exit_policy_sha256"))
        and _only_compact_execution_1m_range_pruning(
            old_inference=old_inference, new_inference=new_inference,
        )
    )
    if old_economics != new_economics:
        raise ValueError("execution economics or policy changed during successor migration")
    model_promotion = (
        allow_approved_model_promotion
        and str(new_inference.get("admission_contract"))
        == "strict_r3_bcf_current_dual_mc1_authority_v1"
        and str(new_execution.get("exit_policy_sha256"))
        == str(old_execution.get("exit_policy_sha256"))
    )
    if (
        not bounded_extension
        and not model_promotion
        and not policy_promotion
        and not friction_revision
        and not exit_vwap_stop_revision
        and not admission_only_execution_authority
        and not execution_adjusted_ev_30bps_gate
        and not close_based_hard_stop_monitor
        and not executable_vwap_frozen_threshold_sentinel
        and not openpositions_503_fills_fallback
        and not direct_15m_book_guard
        and not oi_funding_sidecar_containment
        and not compact_execution_1m_range_pruning
        and _inference_semantics(old_inference) != _inference_semantics(new_inference)
    ):
        raise ValueError("frozen inference/model/admission semantics changed")

    output = copy.deepcopy(state)
    positions_before = copy.deepcopy(output.get("positions") or [])
    decisions_before = copy.deepcopy(output.get("processed_decision_ids") or [])
    output.update({
        "inference_bundle_sha256": new_execution["inference_bundle_sha256"],
        "exit_policy_sha256": new_execution["exit_policy_sha256"],
        "activation_authorization_sha256": new_execution[
            "activation_authorization_sha256"
        ],
        "contract_migration": {
            "kind": (
                APPROVED_POLICY_PROMOTION_KIND if policy_promotion
                else APPROVED_MODEL_PROMOTION_KIND if model_promotion
                else APPROVED_EXECUTION_FRICTION_REVISION_KIND if friction_revision
                else APPROVED_EXIT_VWAP_STOP_PROTECTION_KIND if exit_vwap_stop_revision
                else APPROVED_DUAL_ADMISSION_ONLY_EXECUTION_AUTHORITY_KIND if admission_only_execution_authority
                else APPROVED_EXECUTION_ADJUSTED_EV_30BPS_GATE_KIND if execution_adjusted_ev_30bps_gate
                else APPROVED_CLOSE_BASED_HARD_STOP_MONITOR_KIND if close_based_hard_stop_monitor
                else APPROVED_EXECUTABLE_VWAP_FROZEN_THRESHOLD_SENTINEL_KIND if executable_vwap_frozen_threshold_sentinel
                else APPROVED_OPENPOSITIONS_503_FILLS_FALLBACK_KIND if openpositions_503_fills_fallback
                else APPROVED_DIRECT_15M_BOOK_GUARD_KIND if direct_15m_book_guard
                else APPROVED_OI_FUNDING_SIDECAR_CONTAINMENT_KIND if oi_funding_sidecar_containment
                else "approved_compact_execution_1m_range_pruning_v1" if compact_execution_1m_range_pruning
                else BOUNDED_FRESHNESS_EXTENSION_KIND if bounded_extension
                else MIGRATION_KIND
            ),
            "previous_inference_bundle_sha256": old_execution[
                "inference_bundle_sha256"
            ],
            "new_inference_bundle_sha256": new_execution[
                "inference_bundle_sha256"
            ],
            "positions_preserved_exact": not policy_promotion,
            "processed_decisions_preserved_exact": True,
        },
    })
    if policy_promotion:
        for position in output.get("positions") or []:
            if not isinstance(position, dict):
                continue
            atr = position.get("entry_signal_atr", position.get("atr"))
            if atr is None:
                raise ValueError("open position lacks immutable signal ATR")
            position.setdefault("entry_signal_atr", atr)
            position["atr_source"] = "raw_decision_time_atr"
            position.setdefault("mfe", position.get("maximum_favourable", 0.0))
            position.setdefault("smooth_armed", False)
            position.setdefault("smooth_lock", None)
            position.setdefault("smooth_lock_price", None)
            position.setdefault("last_processed_completed_1m_bar", position.get("next_bar_ts"))
    if not policy_promotion and output.get("positions") != positions_before:
        raise AssertionError("successor migration changed live positions")
    if output.get("processed_decision_ids") != decisions_before:
        raise AssertionError("successor migration changed processed decisions")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--old-execution", type=Path, required=True)
    parser.add_argument("--new-execution", type=Path, required=True)
    parser.add_argument("--out-state", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    parser.add_argument(
        "--allow-bounded-current-hour-freshness-extension",
        action="store_true",
        help=(
            "Allow only the audited 900→3600-second current-hour freshness "
            "extension; every model, feature, policy, economics, and admission "
            "field must remain identical."
        ),
    )
    parser.add_argument(
        "--allow-approved-exit-vwap-stop-protection", action="store_true",
        help=(
            "Permit only the explicit policy-targeted full-size exit-VWAP "
            "protective-stop adjustment; model, admission, portfolio and "
            "parent-policy semantics must remain unchanged."
        ),
    )
    parser.add_argument(
        "--allow-approved-close-based-hard-stop-monitor", action="store_true",
        help=(
            "Permit only the explicitly approved completed-minute close-based "
            "initial hard-stop controller; the full-VWAP controller must remain "
            "disabled and all model, admission, portfolio, native-stop and "
            "parent-policy fields must remain identical."
        ),
    )
    parser.add_argument(
        "--allow-approved-executable-vwap-frozen-threshold-sentinel",
        action="store_true",
        help=(
            "Permit only the explicitly approved 30-second remaining-size "
            "executable-VWAP exit sentinel against the prior completed-1m "
            "threshold, with the native last stop retained as the sealed "
            "lower catastrophe backstop."
        ),
    )
    parser.add_argument(
        "--allow-approved-openpositions-503-fills-fallback",
        action="store_true",
        help=(
            "Permit only the explicitly approved minute-monitor availability "
            "fallback from Kraken /openpositions 503 to its authenticated "
            "/fills ledger for already-tracked positions."
        ),
    )
    parser.add_argument(
        "--allow-approved-direct-15m-book-guard", action="store_true",
        help=(
            "Permit only the reviewed target-free eligibility tightening that "
            "rejects a zero/unknown-volume direct 15-minute decision open when "
            "its contemporaneous book differs by more than 100 bps."
        ),
    )
    parser.add_argument(
        "--allow-approved-oi-funding-sidecar-containment", action="store_true",
        help=(
            "Permit only the reviewed corrupt-local OI/funding sidecar quarantine "
            "repair; it may fail one symbol closed but cannot alter models, policy, "
            "admission, portfolio, or execution economics."
        ),
    )
    parser.add_argument(
        "--allow-approved-compact-execution-1m-range-pruning", action="store_true",
        help=(
            "Permit only recognition of unambiguous legacy compact one-minute "
            "part bounds so a bounded live exit read never decodes unrelated "
            "historical cache years."
        ),
    )
    parser.add_argument(
        "--allow-approved-policy-promotion", action="store_true",
        help=(
            "Permit only the explicitly approved, hash-bound smooth-capital-"
            "protection parent-policy successor; model/admission/portfolio "
            "semantics must remain identical."
        ),
    )
    parser.add_argument(
        "--allow-approved-model-promotion", action="store_true",
        help=(
            "Permit an explicitly approved BCF/current dual-admission promotion "
            "only when the exit-policy hash and execution economics are unchanged; "
            "open positions and processed decisions remain byte-identical."
        ),
    )
    parser.add_argument(
        "--allow-approved-execution-friction-revision", action="store_true",
        help=(
            "Permit only the explicit live entry-friction revision from the "
            "80-bps floor to book-derived friction plus the sealed 10-bps buffer; "
            "all model, policy, admission, and portfolio semantics must match."
        ),
    )
    parser.add_argument(
        "--allow-approved-dual-admission-only-execution-authority",
        action="store_true",
        help=(
            "Permit the explicit user-approved change that makes sealed dual "
            "MC1 admission plus the common portfolio auction the sole entry "
            "authority; order-book values remain telemetry/protection-only."
        ),
    )
    parser.add_argument(
        "--allow-approved-execution-adjusted-ev-30bps-gate",
        action="store_true",
        help=(
            "Permit only the explicit user-approved restoration of the final "
            "+30-bps execution-adjusted-EV safety veto after the sealed dual "
            "MC1/common-auction selection; it may not rerank candidates."
        ),
    )
    args = parser.parse_args()
    if args.out_state.exists() or args.out_receipt.exists():
        raise FileExistsError("successor migration outputs are immutable")

    state = json.loads(args.state.read_text())
    old_execution = _resolve_execution_payload(args.old_execution)
    new_execution = _resolve_execution_payload(args.new_execution)
    old_inference_path = ROOT / old_execution["inference_bundle"]
    new_inference_path = ROOT / new_execution["inference_bundle"]
    if _sha(old_inference_path) != old_execution["inference_bundle_sha256"]:
        raise ValueError("old inference bundle file hash mismatch")
    _validate_successor_live_authorization(new_execution)
    from extreme_price_movements.strict_r3_inference_bundle import StrictR3InferenceBundle

    # A promoted inference contract can be a narrow, hash-bound overlay.  Use
    # the same resolved payload as scoring rather than mistaking the overlay's
    # small JSON wrapper for its complete model/admission contract.
    new_inference = dict(
        StrictR3InferenceBundle.load(new_inference_path, root=ROOT).payload
    )
    old_inference = _resolve_overlay_semantics_without_runtime_load(
        old_inference_path
    )
    output = migrate_runtime_successor_state(
        state=state,
        old_execution=old_execution,
        new_execution=new_execution,
        old_inference=old_inference,
        new_inference=new_inference,
        allow_bounded_current_hour_extension=(
            args.allow_bounded_current_hour_freshness_extension
        ),
        allow_approved_model_promotion=args.allow_approved_model_promotion,
        allow_approved_policy_promotion=args.allow_approved_policy_promotion,
        allow_approved_execution_friction_revision=(
            args.allow_approved_execution_friction_revision
        ),
        allow_approved_exit_vwap_stop_protection=(
            args.allow_approved_exit_vwap_stop_protection
        ),
        allow_approved_close_based_hard_stop_monitor=(
            args.allow_approved_close_based_hard_stop_monitor
        ),
        allow_approved_executable_vwap_frozen_threshold_sentinel=(
            args.allow_approved_executable_vwap_frozen_threshold_sentinel
        ),
        allow_approved_dual_admission_only_execution_authority=(
            args.allow_approved_dual_admission_only_execution_authority
        ),
        allow_approved_execution_adjusted_ev_30bps_gate=(
            args.allow_approved_execution_adjusted_ev_30bps_gate
        ),
        allow_approved_openpositions_503_fills_fallback=(
            args.allow_approved_openpositions_503_fills_fallback
        ),
        allow_approved_direct_15m_book_guard=(
            args.allow_approved_direct_15m_book_guard
        ),
        allow_approved_oi_funding_sidecar_containment=(
            args.allow_approved_oi_funding_sidecar_containment
        ),
        allow_approved_compact_execution_1m_range_pruning=(
            args.allow_approved_compact_execution_1m_range_pruning
        ),
    )
    atomic_json(args.out_state, output)
    receipt = {
        "schema": "strict_r3_live_state_runtime_successor_migration_v1",
        "migration_kind": str(output.get("contract_migration", {}).get("kind")),
        "source_state": str(args.state),
        "source_state_sha256": _sha(args.state),
        "output_state": str(args.out_state),
        "output_state_sha256": _sha(args.out_state),
        "old_execution": str(args.old_execution),
        "old_execution_sha256": _sha(args.old_execution),
        "new_execution": str(args.new_execution),
        "new_execution_sha256": _sha(args.new_execution),
        "positions_preserved_exact": output.get("positions") == state.get("positions"),
        "positions_policy_state_initialized": bool(
            args.allow_approved_policy_promotion
        ),
        "processed_decisions_preserved_exact": (
            output.get("processed_decision_ids")
            == state.get("processed_decision_ids")
        ),
        "position_count": len(output.get("positions") or []),
    }
    atomic_json(args.out_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
