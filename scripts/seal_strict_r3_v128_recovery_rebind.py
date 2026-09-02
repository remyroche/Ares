#!/usr/bin/env python3
"""Seal the one-path recovery verifier rebind for the strict-R3 v153 runtime.

The state payload is unchanged.  The sole new runtime behavior is a narrowly
bounded verifier that permits the already-audited panel-updater receipt to
advance from its prior implementation hash on the first recovered hour.  The
successor cannot alter models, features, Geometry/K9, calibration, admission,
portfolio, or exit economics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v127_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_parallel15m.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v55_"
    "v127_bcf_current_dual_samebundle21d_feature_state_reseal_parallel15m.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v129_v153_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_parallel15m.json"
)
STATE_BUNDLE = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_features_runtime_reseal_"
    "20260822T080000Z_v1/bundle"
)
UPDATER_PARITY = Path(
    "/private/tmp/strict_r3_panel_state_current_source_parity_"
    "20260822T080000Z.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v128_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_recovery_rebind.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v56_"
    "v128_bcf_current_dual_samebundle21d_feature_state_reseal_recovery_rebind.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v130_v154_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_recovery_rebind.json"
)
OUT_REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_bcf_feature_state_reseal_20260822_v4/"
    "recovery_rebind_review.json"
)

RUNNER = "scripts/run_strict_r3_hourly_shadow_resume_v15.py"
UPDATER = "scripts/update_strict_r3_feature_panel_state.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base = read(ROOT / str(overlay["base_bundle"]))
    result = dict(base.get("runtime_code_sha256") or {})
    result.update(dict(overlay.get("overrides", {}).get("runtime_code_sha256") or {}))
    return result


def main() -> None:
    for path in (
        SOURCE_OVERLAY, SOURCE_AUTHORIZATION, SOURCE_EXECUTION,
        STATE_BUNDLE / "state_bundle_manifest.json", UPDATER_PARITY,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    source_overlay = read(SOURCE_OVERLAY)
    source_hashes = resolved_runtime_hashes(source_overlay)
    current_hashes = {relative: sha(ROOT / relative) for relative in source_hashes}
    changed = {
        relative for relative in source_hashes
        if current_hashes[relative] != source_hashes[relative]
    }
    if changed != {RUNNER}:
        raise ValueError(f"unexpected runtime delta: {sorted(changed)}")

    state_manifest = read(STATE_BUNDLE / "state_bundle_manifest.json")
    implementation = dict(state_manifest.get("implementation_sha256") or {})
    updater_prior = str(implementation.get(UPDATER) or "")
    updater_current = sha(ROOT / UPDATER)
    if not updater_prior or updater_prior == updater_current:
        raise ValueError("state receipt does not require the audited updater rebind")
    parity = read(UPDATER_PARITY)
    if parity != {
        "exact": True,
        "fields": 35,
        "mismatch_fields": [],
        "timestamp": "2026-08-22T07:00:00+00:00",
    }:
        raise ValueError("panel-updater append parity is not an exact sealed-row pass")

    overlay = copy.deepcopy(source_overlay)
    override_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    override_hashes.update(current_hashes)
    overlay["overrides"]["runtime_code_sha256"] = override_hashes
    overlay["purpose"] = (
        "v128: focused recovery-verifier rebind. The v153 one-time state "
        "re-receipt remains byte-identical in payload; only the recovery runner "
        "can recognise the independently exact panel-updater implementation "
        "transition for that sealed state. Economics are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNNER],
        "state_implementation_rebind": {
            "source_state_bundle": str(STATE_BUNDLE.relative_to(ROOT)),
            "source_state_manifest_sha256": sha(STATE_BUNDLE / "state_bundle_manifest.json"),
            "path": UPDATER,
            "prior_hash": updater_prior,
            "current_hash": updater_current,
            "exact_append_parity": str(UPDATER_PARITY),
            "exact_append_parity_sha256": sha(UPDATER_PARITY),
        },
        "economic_contract_changed": False,
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(read(SOURCE_AUTHORIZATION))
    authorization.update({
        "authorization_source": (
            "User-approved live continuation after the v153 stateful recovery "
            "verifier was resealed to recognise exactly one independently audited "
            "panel-updater receipt transition; no economic change."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(read(SOURCE_EXECUTION))
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_current = {relative: sha(ROOT / relative) for relative in execution_hashes}
    execution_changed = {
        relative for relative in execution_hashes
        if execution_current[relative] != execution_hashes[relative]
    }
    # The shadow-only recovery runner is sealed by the inference overlay.  It
    # is intentionally not an execution-contract dependency, so the execution
    # runtime namespace itself must remain byte-identical.
    if execution_changed:
        raise ValueError(f"unexpected execution runtime delta: {sorted(execution_changed)}")
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "runtime_code_sha256": execution_current,
        "version_note": (
            "v154: versioned v153 recovery-verifier rebind for the single audited "
            "panel-updater receipt transition. Models, score, Geometry/K9, "
            "admission, portfolio and rich exit are unchanged."
        ),
    })
    predecessors = list(execution.get("runtime_reseal_predecessors") or [])
    predecessors.append({
        "successor_execution_semantics": "bcf_samebundle21d_feature_state_recovery_rebind_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "allowed_runtime_code_paths": [RUNNER],
        "economic_contract_changed": False,
        "reason": (
            "Recover the contiguous state chain using the sealed v153 re-receipt "
            "and its one audited panel-updater implementation transition only."
        ),
    })
    execution["runtime_reseal_predecessors"] = predecessors
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_feature_state_recovery_rebind_review_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "source_authorization": str(SOURCE_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "changed_runtime_paths": [RUNNER],
        "state_implementation_rebind": overlay["runtime_reseal"]["state_implementation_rebind"],
        "economic_contract_changed": False,
        "operator_state_payload_changed": False,
        "validation": {
            "panel_updater_append_exact": True,
            "panel_fields": 35,
            "candidate_or_model_change": False,
            "order_submission_in_recovery": False,
        },
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps({"status": "pass", **review}, sort_keys=True))


if __name__ == "__main__":
    main()
