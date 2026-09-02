#!/usr/bin/env python3
"""Seal the verified same-lineage BCF challenger as a live *candidate*.

This creates contracts only.  It does not start a service, rebuild state,
refresh data, score candidates, or submit an order.  Live activation still
requires a fresh no-order recovery/parity receipt and an explicit successor
launcher.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_v153_v181_hash_stability_capacity_fallback.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v84_v182_shadow_recovery_hash_rebind_live.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v158_v182_shadow_recovery_hash_rebind_live.json"
CHALLENGER = ROOT / "data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_v155_v183_bcf_same_lineage_challenger.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260824_v85_v183_bcf_same_lineage_challenger.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v159_v183_bcf_same_lineage_challenger.json"
OUT_REVIEW = CHALLENGER / "live_candidate_seal" / "run_manifest.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _update_runtime_hashes(overlay: dict) -> None:
    hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in hashes:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        hashes[relative] = sha(source)
    overlay["overrides"]["runtime_code_sha256"] = hashes


def main() -> None:
    bcf_dir = CHALLENGER / "bcf_bundle/bundles/month=2026-08"
    bcf_manifest = bcf_dir / "run_manifest.json"
    mc1_dir = CHALLENGER / "bcf_mc1_bundle"
    mc1_manifest = mc1_dir / "run_manifest.json"
    ledger = CHALLENGER / "bcf_mc1_oos_ledger.parquet"
    reference = CHALLENGER / "features/canonical120_features.parquet"
    policy = ROOT / "config/strict_r3_bcf_current_dual_mc1_portfolio_challenger_v1.json"
    required = [bcf_dir / "monthly_bundle.joblib", bcf_manifest, mc1_dir / "bcf_mc1_d2.joblib", mc1_manifest, ledger, reference, policy]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    bcf = load(bcf_manifest)
    mapper = load(mc1_manifest)
    root_bcf_manifest = load(CHALLENGER / "bcf_bundle/run_manifest.json")
    if root_bcf_manifest.get("status") != "complete" or bcf.get("side_name") != "long":
        raise ValueError("BCF challenger bundle is incomplete or wrong-side")
    if str(bcf.get("bundle_sha256") or "") != sha(bcf_dir / "monthly_bundle.joblib"):
        raise ValueError("BCF challenger monthly bundle hash mismatch")
    if bcf.get("cutoff") != "2026-08-01T00:00:00+00:00":
        raise ValueError("BCF challenger cutoff does not match the August producer")
    if mapper.get("contract") != "strict_r3_bcf_mc1_d2_authority_v1":
        raise ValueError("BCF MC1 challenger contract mismatch")
    if float(mapper.get("admission_threshold_bps", float("nan"))) != 30.0:
        raise ValueError("BCF MC1 challenger threshold changed")

    source_overlay = load(SOURCE_OVERLAY)
    overlay = copy.deepcopy(source_overlay)
    overrides = overlay["overrides"]
    paths = overrides["paths"]
    hashes = overrides["sha256"]
    replacement_paths = {
        "bcf_monthly_bundle_dir": bcf_dir,
        "bcf_reference_ledger": reference,
        "bcf_mc1_bundle_dir": mc1_dir,
        "bcf_mc1_ledger": ledger,
        "dual_portfolio_policy": policy,
    }
    hash_keys = {
        "bcf_monthly_bundle_dir": "dual_bcf_monthly_bundle",
        "bcf_reference_ledger": "dual_bcf_reference_ledger",
        "bcf_mc1_bundle_dir": "dual_bcf_mc1_model",
        "bcf_mc1_ledger": "dual_bcf_mc1_ledger",
        "dual_portfolio_policy": "dual_portfolio_policy",
    }
    for key, path in replacement_paths.items():
        paths[key] = str(path.relative_to(ROOT))
        if key == "bcf_monthly_bundle_dir":
            hashes["dual_bcf_monthly_bundle"] = sha(path / "monthly_bundle.joblib")
            hashes["dual_bcf_monthly_manifest"] = sha(path / "run_manifest.json")
        elif key == "bcf_mc1_bundle_dir":
            hashes["dual_bcf_mc1_model"] = sha(path / "bcf_mc1_d2.joblib")
            hashes["dual_bcf_mc1_manifest"] = sha(path / "run_manifest.json")
        else:
            hashes[hash_keys[key]] = sha(path)
    _update_runtime_hashes(overlay)
    overlay["purpose"] = (
        "v183 live candidate: replace the unavailable historical August BCF serialization "
        "with the independently parity-checked same-lineage BCF base/consensus/Severe/MC1 artifacts. "
        "This file grants no entry authority without a fresh recovery and successor activation."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_bcf_same_lineage_candidate_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_artifacts": [str(path.relative_to(ROOT)) for path in required[:-1]],
        "reason": "rebuild unavailable BCF model from one target-free causal feature lineage",
        "live_authority": "blocked_pending_fresh_state_recovery_and_parity",
    }
    write(OUT_OVERLAY, overlay)

    auth = copy.deepcopy(load(SOURCE_AUTH))
    auth.update({
        "authorization_source": "User-authorized launch of the verified BCF same-lineage challenger after recovery/parity/reseal.",
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_pending": [
            "fresh target-free challenger score",
            "independent dual-stack parity",
            "no-order recovered state chain",
            "fresh successor runtime checkpoint",
        ],
    })
    write(OUT_AUTH, auth)

    execution = copy.deepcopy(load(SOURCE_EXECUTION))
    execution["overrides"].update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
    })
    write(OUT_EXECUTION, execution)
    review = {
        "schema": "strict_r3_bcf_same_lineage_live_candidate_seal_v1",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation": str(OUT_AUTH.relative_to(ROOT)),
        "activation_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
        "bcf_bundle_sha256": sha(bcf_dir / "monthly_bundle.joblib"),
        "bcf_mc1_model_sha256": sha(mc1_dir / "bcf_mc1_d2.joblib"),
        "bcf_mc1_ledger_sha256": sha(ledger),
        "fresh_state_recovery_required": True,
        "live_service_started": False,
        "exchange_calls": 0,
        "order_submission": False,
    }
    write(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
