#!/usr/bin/env python3
"""Seal the BCF MC1 same-bundle 21-day replay-ledger live successor."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v122_bcf_current_dual_bcf_mc1_structural_prior_coldstart.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260822_v50_v122_bcf_mc1_structural_prior_coldstart.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v124_v148_bcf_mc1_structural_prior_coldstart.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v123_bcf_current_dual_bcf_mc1_samebundle21d_replay.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260822_v51_v123_bcf_mc1_samebundle21d_replay.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v125_v149_bcf_mc1_samebundle21d_replay.json"
OUT_REVIEW = ROOT / "data_perp/artifacts/strict_r3_bcf_mc1_samebundle21d_replay_reseal_20260822_v1/runtime_review.json"
LEDGER = ROOT / "data_perp/artifacts/strict_r3_bcf_same_bundle_recent_replay_ledger_20260822T090000Z_v1/bcf_mc1_recent_replay_ledger.parquet"
LEDGER_MANIFEST = LEDGER.parent / "run_manifest.json"
MAPPER = ROOT / "extreme_price_movements/strict_r3_bcf_mc1_mapper.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _static(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("purpose", None)
    value.pop("runtime_reseal", None)
    overrides = value.get("overrides") or {}
    overrides.pop("runtime_code_sha256", None)
    paths = dict(overrides.get("paths") or {})
    paths.pop("bcf_mc1_ledger", None)
    overrides["paths"] = paths
    hashes = dict(overrides.get("sha256") or {})
    hashes.pop("dual_bcf_mc1_ledger", None)
    overrides["sha256"] = hashes
    return value


def main() -> None:
    if not LEDGER.is_file() or not LEDGER_MANIFEST.is_file():
        raise FileNotFoundError("same-bundle BCF replay ledger is incomplete")
    ledger_manifest = json.loads(LEDGER_MANIFEST.read_text())
    if ledger_manifest.get("schema") != "strict_r3_bcf_same_bundle_recent_replay_ledger_v1":
        raise ValueError("unexpected BCF replay ledger schema")
    if int(ledger_manifest.get("resolved_hours", 0)) < 400:
        raise ValueError("BCF replay ledger lacks sufficient resolved hourly support")
    if str(ledger_manifest.get("ledger_sha256")) != sha(LEDGER):
        raise ValueError("BCF replay ledger hash mismatch")

    source = json.loads(SOURCE_OVERLAY.read_text())
    overlay = copy.deepcopy(source)
    overrides = overlay["overrides"]
    overrides["paths"]["bcf_mc1_ledger"] = str(LEDGER.relative_to(ROOT))
    overrides["sha256"]["dual_bcf_mc1_ledger"] = sha(LEDGER)
    overrides["runtime_code_sha256"][str(MAPPER.relative_to(ROOT))] = sha(MAPPER)
    overlay["purpose"] = (
        "v123: BCF MC1 uses the frozen August BCF bundle's same-bundle, target-free "
        "21-day score replay joined after scoring to strictly prior resolved parent-policy "
        "trades. No recent same-bundle support is fail-closed; no zero-shift fallback is used."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [str(MAPPER.relative_to(ROOT))],
        "changed_calibration_artifacts": [str(LEDGER.relative_to(ROOT))],
        "reason": "Replace stale BCF history with causal same-bundle 21-day replay support.",
    }
    if _static(source) != _static(overlay):
        raise AssertionError("successor changed a non-calibration static contract")
    write_new(OUT_OVERLAY, overlay)

    auth = copy.deepcopy(json.loads(SOURCE_AUTH.read_text()))
    auth.update({
        "authorization_source": "User-approved BCF MC1 same-bundle 21-day replay calibration repair.",
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, auth)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "version_note": "v149: BCF same-bundle 21-day replay calibration ledger; no zero-shift cold-start fallback.",
    })
    execution["runtime_reseal_predecessors"] = list(execution.get("runtime_reseal_predecessors") or []) + [{
        "successor_execution_semantics": "bcf_mc1_samebundle_21d_replay_calibration_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "approved_calibration_artifact": str(LEDGER.relative_to(ROOT)),
        "approved_calibration_artifact_sha256": sha(LEDGER),
        "allowed_runtime_code_paths": [str(MAPPER.relative_to(ROOT))],
        "reviewed_current_runtime": True,
    }]
    write_new(OUT_EXECUTION, execution)
    review = {
        "schema": "strict_r3_bcf_mc1_samebundle21d_replay_reseal_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTH),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "mapper_sha256": sha(MAPPER),
        "replay_ledger": str(LEDGER.relative_to(ROOT)),
        "replay_ledger_sha256": sha(LEDGER),
        "replay_manifest": str(LEDGER_MANIFEST.relative_to(ROOT)),
        "replay_rows": int(ledger_manifest["rows"]),
        "resolved_hours": int(ledger_manifest["resolved_hours"]),
        "missing_score_hours": ledger_manifest["missing_score_hours"],
        "non_calibration_static_contract_exact": _static(source) == _static(overlay),
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
