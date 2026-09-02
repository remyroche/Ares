#!/usr/bin/env python3
"""Seal a candidate-only BCF challenger over the verified no-refit geometry recovery.

This deliberately combines two already separately audited changes: the
same-lineage August BCF challenger and the embedded, output-identical
Geometry/K9 identity recovery.  It grants no live-entry authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v164_v193_historical_challenger_upstream_recovery_candidate.json"
GEOMETRY_SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v158_v187_geometry_universe_exact1m_parent_policy_rebind.json"
OUT = ROOT / "config/strict_r3_inference_overlay_long_v165_v194_historical_dual_challenger_recovery_candidate.json"
RECEIPT = ROOT / "data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/historical_dual_challenger_recovery_candidate/run_manifest.json"
PARENT_REBIND_ROOT = ROOT / "data_perp/artifacts/strict_r3_parent_policy_calibration_ledger_rebind_20260824_v1"
CALIBRATION_POLICY = PARENT_REBIND_ROOT / "parent_policy_semantic_rebind.json"
HISTORICAL_BCF_MC1_ROOT = ROOT / "data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/bcf_mc1_historical_commonpolicy_cutoff_20260801"
HISTORICAL_BCF_MC1_LEDGER = ROOT / "data_perp/artifacts/strict_r3_score_family_common_policy_intersection_20260816_v1/bcf_common_policy_ledger.parquet"
UPSTREAM_CHALLENGER_DIR = ROOT / "data_perp/artifacts/strict_r3_lockstep_successor28_long_aug1_7_current_spread_20260812_v1/bundles/cutoff=20260801/upstream"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError("candidate outputs already exist")
    candidate = copy.deepcopy(_load(SOURCE))
    geometry = _load(GEOMETRY_SOURCE)
    paths = candidate["overrides"]["paths"]
    hashes = candidate["overrides"]["sha256"]
    geometry_paths = geometry["overrides"]["paths"]
    geometry_hashes = geometry["overrides"]["sha256"]
    for key in ("frozen_geometry_bundle", "frozen_universe_manifest", "exit_policy"):
        paths[key] = geometry_paths[key]
        hashes[key] = geometry_hashes[key]
    # The original homogeneous-28 August upstream serialization was removed
    # (the surviving path is zero bytes).  This recovery candidate therefore
    # binds the documented, non-empty August lock-step upstream peer as a
    # *challenger*, never as byte-identical restoration of the deleted model.
    upstream_bundle = UPSTREAM_CHALLENGER_DIR / "monthly_upstream_bundle.joblib"
    upstream_manifest = UPSTREAM_CHALLENGER_DIR / "run_manifest.json"
    if not upstream_bundle.is_file() or upstream_bundle.stat().st_size == 0:
        raise FileNotFoundError("non-empty challenger upstream bundle is unavailable")
    paths["upstream_bundle_dir"] = str(UPSTREAM_CHALLENGER_DIR.relative_to(ROOT))
    hashes["upstream_bundle"] = _sha(upstream_bundle)
    hashes["upstream_manifest"] = _sha(upstream_manifest)
    # MC1 and the resolved calibration ledger were fitted to the source-aligned
    # parent policy.  The rich policy remains the exit-only overlay.  Binding
    # both prevents the runtime ledger from accidentally treating the rich
    # execution policy as the calibration target.
    paths["calibration_policy"] = str(CALIBRATION_POLICY.relative_to(ROOT))
    hashes["calibration_policy"] = _sha(CALIBRATION_POLICY)
    paths["resolved_score_label_ledger"] = str(
        (PARENT_REBIND_ROOT / "walkforward_scored_label_ledger.parquet").relative_to(ROOT)
    )
    hashes["resolved_score_label_ledger"] = _sha(
        PARENT_REBIND_ROOT / "walkforward_scored_label_ledger.parquet"
    )
    # The first same-lineage BCF MC1 challenger has a 2026-08-24 fit cutoff
    # and must never be used to replay earlier decisions.  This candidate
    # instead binds the separately trained common-policy map whose cutoff is
    # 2026-08-01 and whose training labels are strictly resolved OOS rows.
    if not (HISTORICAL_BCF_MC1_ROOT / "bcf_mc1_d2.joblib").is_file() or HISTORICAL_BCF_MC1_LEDGER.stat().st_size == 0:
        raise FileNotFoundError("historical BCF MC1 challenger inputs are unavailable")
    paths["bcf_mc1_bundle_dir"] = str(HISTORICAL_BCF_MC1_ROOT.relative_to(ROOT))
    paths["bcf_mc1_ledger"] = str(HISTORICAL_BCF_MC1_LEDGER.relative_to(ROOT))
    hashes["dual_bcf_mc1_model"] = _sha(HISTORICAL_BCF_MC1_ROOT / "bcf_mc1_d2.joblib")
    hashes["dual_bcf_mc1_manifest"] = _sha(HISTORICAL_BCF_MC1_ROOT / "run_manifest.json")
    hashes["dual_bcf_mc1_ledger"] = _sha(HISTORICAL_BCF_MC1_LEDGER)
    runtime_hashes = candidate["overrides"].setdefault("runtime_code_sha256", {})
    for relative in (
        "scripts/run_strict_r3_shadow_cycle.py",
        "scripts/assemble_strict_r3_runtime_resolved_ledger.py",
        "scripts/run_strict_r3_hourly_shadow_resume_v15.py",
        "extreme_price_movements/strict_r3_inference_bundle.py",
    ):
        runtime_hashes[relative] = _sha(ROOT / relative)
    hashes["frozen_geometry_manifest"] = geometry_hashes["frozen_geometry_manifest"]
    candidate["purpose"] = (
        "v194 historical-recovery-only dual challenger: same-lineage August BCF challenger combined with the "
        "sealed conversion-bundle embedded Geometry/K9 recovery and recovered universe "
        "witness. No model or geometry is refit; live authority remains blocked pending "
        "fresh state recovery and independent parity.  The source-aligned parent "
        "policy is hash-bound for MC1/calibration; the rich policy remains exit-only. "
        "BCF MC1 uses a separately trained 2026-08-01-cutoff challenger fit only on strictly OOS "
        "scores and resolved parent-policy labels. The deleted homogeneous-28 "
        "upstream serialization is explicitly replaced by the documented non-empty current-spread "
        "August lock-step upstream challenger; this is not canonical byte restoration."
    )
    candidate["runtime_reseal"] = {
        "schema": "strict_r3_historical_dual_challenger_recovery_candidate_v1",
        "supersedes": [str(SOURCE.relative_to(ROOT)), str(GEOMETRY_SOURCE.relative_to(ROOT))],
        "economic_contract_changed": False,
        "calibration_policy_role": "source_aligned_parent_labels_for_current_v5_MC1",
        "execution_policy_role": "rich_exit_only",
        "parent_ledger_rebind": "byte_preserving_semantic_policy_rebind",
        "same_day_policy_identity_transition": "reset_to_sealed_base_no_current_labels",
        "geometry_refit": False,
        "semantic_geometry_parent_sha256": "ad7eae631da909feddee7349d07fd8ef377db173067d971bee33d24d82f20eb4",
        "semantic_geometry_view_sha256": "dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638",
        "bcf_retrain_status": "same_lineage_challenger_score_with_pre_august_common_policy_mc1",
        "upstream_retrain_status": "documented_august_lockstep_peer_after_deleted_original",
        "deleted_original_upstream_sha256": "8d8139b166dc0af69815247e2abab1999a6670b0d7cb5552a485dd7ea0006a0e",
        "challenger_upstream_sha256": hashes["upstream_bundle"],
        "canonical_byte_parity": False,
        "bcf_mc1_fit_cutoff": "2026-08-01T00:00:00+00:00",
        "live_authority": "blocked_pending_fresh_no_order_state_recovery_and_independent_parity",
    }
    _write_new(OUT, candidate)
    receipt = {
        "schema": "strict_r3_historical_dual_challenger_recovery_candidate_v1",
        "status": "candidate_only",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "inference_bundle": str(OUT.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OUT),
        "source_bcf_candidate": str(SOURCE.relative_to(ROOT)),
        "source_geometry_rebind": str(GEOMETRY_SOURCE.relative_to(ROOT)),
        "challenger_upstream_bundle": str(upstream_bundle.relative_to(ROOT)),
        "challenger_upstream_bundle_sha256": _sha(upstream_bundle),
        "deleted_original_upstream_sha256": "8d8139b166dc0af69815247e2abab1999a6670b0d7cb5552a485dd7ea0006a0e",
        "canonical_byte_parity": False,
        "geometry_refit": False,
        "economic_contract_changed": False,
        "calibration_policy": str(CALIBRATION_POLICY.relative_to(ROOT)),
        "calibration_policy_sha256": _sha(CALIBRATION_POLICY),
        "parent_ledger_rebind_manifest": str((PARENT_REBIND_ROOT / "run_manifest.json").relative_to(ROOT)),
        "parent_ledger_rebind_manifest_sha256": _sha(PARENT_REBIND_ROOT / "run_manifest.json"),
        "historical_bcf_mc1_bundle": str(HISTORICAL_BCF_MC1_ROOT.relative_to(ROOT)),
        "historical_bcf_mc1_bundle_sha256": _sha(HISTORICAL_BCF_MC1_ROOT / "bcf_mc1_d2.joblib"),
        "historical_bcf_mc1_ledger": str(HISTORICAL_BCF_MC1_LEDGER.relative_to(ROOT)),
        "historical_bcf_mc1_ledger_sha256": _sha(HISTORICAL_BCF_MC1_LEDGER),
        "exchange_calls": 0,
        "order_submission": False,
        "required_before_live_activation": [
            "bundle_validation",
            "fresh_no_order_state_recovery",
            "independent_feature_score_admission_parity",
            "fresh_successor_runtime_checkpoint",
        ],
    }
    _write_new(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
