#!/usr/bin/env python3
"""Seal the reviewed newest-source successor of the dual BCF/current stack."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v67_feature_runtime_parity_candidate.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v65_v75_bcf_current_dual_cached_parallel_dualauction.json"
SOURCE_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v30_bcf_current_dual_cached_parallel_dualauction.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v80_bcf_current_dual_newest_feature_runtime_exact_stream_audit.json"
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v78_v100_bcf_current_dual_newest_feature_runtime_exact_stream_audit.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v08_bcf_current_dual_newest_feature_runtime_exact_stream_audit.json"
PREDECESSOR_OVERLAYS = (
    ROOT / "config/strict_r3_inference_overlay_long_20260801_v60_bcf_current_dual_mc1.json",
    ROOT / "config/strict_r3_inference_overlay_long_20260801_v65_bcf_current_dual_mc1_cached_parallel.json",
)
STATE_RESEAL_OLD = ROOT / "data_perp/artifacts/strict_r3_successor_v60_live_20260817T180000Z_v1/feature_state/bundle"
STATE_RESEAL_NEW = ROOT / "data_perp/artifacts/strict_r3_v84_v60_current_code_feature_state_successor_20260817T180000Z_v1/bundle"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolved_runtime_hashes(path: Path) -> dict[str, str]:
    """Resolve only sealed hash declarations; never load executable code."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_inference_bundle_overlay_v1":
        return dict(payload["runtime_code_sha256"])
    base = json.loads((ROOT / payload["base_bundle"]).read_text())
    hashes = dict(base["runtime_code_sha256"])
    hashes.update(dict(payload["overrides"].get("runtime_code_sha256") or {}))
    return hashes


def operator_state_payload_sha(bundle: Path) -> str:
    inventory = pd.read_parquet(bundle / "operator_state_inventory.parquet")
    digest = hashlib.sha256()
    for row in inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    ).itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    overlay = copy.deepcopy(json.loads(SOURCE_OVERLAY.read_text()))
    overlay["purpose"] = (
        "v77: user-approved newest runtime source reseal. The frozen models, "
        "feature contract, admission, BCF/current dual-MC1 authority, portfolio "
        "auction, and rich exit policy are unchanged; only reviewed implementation "
        "hashes advance."
    )
    runtime_hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    runtime_hashes[
        "scripts/assemble_strict_r3_stateful_successor_prefix.py"
    ] = ""
    overlay["overrides"]["runtime_code_sha256"] = {
        relative: sha(ROOT / relative)
        for relative in runtime_hashes
    }
    state_payload_sha = operator_state_payload_sha(STATE_RESEAL_OLD)
    if state_payload_sha != operator_state_payload_sha(STATE_RESEAL_NEW):
        raise ValueError("one-time state reseal changed an operator-state payload")
    runtime = overlay["overrides"].setdefault("runtime", {})
    base_payload = json.loads((ROOT / overlay["base_bundle"]).read_text())
    feature_state = dict(base_payload["runtime"]["feature_state"])
    feature_state["one_time_state_reseal"] = {
        "superseded_bundle": str(STATE_RESEAL_OLD.relative_to(ROOT)),
        "resealed_bundle": str(STATE_RESEAL_NEW.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(
            STATE_RESEAL_OLD / "state_bundle_manifest.json"
        ),
        "resealed_manifest_sha256": sha(
            STATE_RESEAL_NEW / "state_bundle_manifest.json"
        ),
        "operator_state_payload_sha256": state_payload_sha,
        "reason": (
            "One-time, hash-bound newest-code state re-receipt; persisted "
            "operator payloads are byte-identical to the exact v60 predecessor."
        ),
    }
    runtime["feature_state"] = feature_state
    write_new(OVERLAY, overlay)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "version_note": (
            "v77: newest-source runtime successor of the approved BCF/current "
            "dual-MC1 live stack; no model, feature contract, admission, portfolio, "
            "entry-economics, or rich-exit parameter change."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "runtime_code_sha256": {
            relative: sha(ROOT / relative)
            for relative in dict(execution["runtime_code_sha256"])
        },
    })
    # This is an explicit, narrow state-transition bridge: it permits only
    # the reviewed newest runtime sources to consume the predecessor's
    # append-only feature/Geometry/K9/portfolio state.  Every non-runtime
    # model, policy and data-contract field remains byte-identical.
    current_runtime = resolved_runtime_hashes(OVERLAY)
    bridges = []
    for predecessor_overlay in PREDECESSOR_OVERLAYS:
        prior_runtime = resolved_runtime_hashes(predecessor_overlay)
        bridges.append({
            "predecessor_inference_bundle": str(predecessor_overlay.relative_to(ROOT)),
            "predecessor_inference_bundle_sha256": sha(predecessor_overlay),
            "current_inference_bundle_sha256": sha(OVERLAY),
            "allowed_runtime_code_paths": sorted(
                key for key in set(prior_runtime) & set(current_runtime)
                if str(prior_runtime[key]) != str(current_runtime[key])
            ),
            "added_runtime_code_paths": sorted(set(current_runtime) - set(prior_runtime)),
            "reason": (
                "User-approved newest code transition; static strategy, model, "
                "feature, admission, portfolio and exit contracts are unchanged."
            ),
        })
    execution["runtime_reseal_predecessors"] = bridges

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-approved newest-code successor: retain the frozen BCF/current "
            "dual-MC1 strategy and reseal reviewed runtime sources only."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "exit_policy": execution["exit_policy"],
        "exit_policy_sha256": execution["exit_policy_sha256"],
    })
    write_new(AUTHORIZATION, authorization)
    execution["activation_authorization_sha256"] = sha(AUTHORIZATION)
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY.relative_to(ROOT)), "overlay_sha256": sha(OVERLAY),
        "execution": str(EXECUTION.relative_to(ROOT)), "execution_sha256": sha(EXECUTION),
        "authorization": str(AUTHORIZATION.relative_to(ROOT)), "authorization_sha256": sha(AUTHORIZATION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
