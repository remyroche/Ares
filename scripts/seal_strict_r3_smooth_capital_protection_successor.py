#!/usr/bin/env python3
"""Seal the explicitly approved smooth-capital-protection live successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1/frozen_challenger.json"
BASE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v58_v68_bcf_current_dual_rich_policy.json"
BASE_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v23_bcf_current_dual.json"
POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
MANIFEST = POLICY.with_name("run_manifest.json")
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v59_v69_bcf_current_dual_smooth_protection.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v24_bcf_current_dual_smooth_protection.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    policy = copy.deepcopy(json.loads(BASE_POLICY.read_text()))
    params = dict(policy["params"])
    # Smooth protection supersedes the previous discontinuous capital-protect
    # branch in this successor.  Trailing and fast-adverse HPO parameters are
    # unchanged.
    params.update({
        "capital_protect_mfe_mult": 0.0,
        "capital_protect_lock_frac": None,
        "capital_protect_min_lock_bps": 0.0,
        "smooth_capital_protection_enabled": True,
        "protection_unit": "raw_decision_time_atr",
        "protection_activation_atr": 1.5,
        "protection_strength": 0.5,
        "protection_power": 1.5,
    })
    policy["params"] = params
    policy["schema"] = "strict_r3_rich_simple_policy_challenger_v1"
    policy["version_note"] = (
        "v69 live successor: immutable raw decision-time ATR smooth capital "
        "protection; hard stop -> prior smooth lock -> trailing -> fast adverse "
        "-> next-bar MFE/arm updates."
    )
    write_new(POLICY, policy)
    policy_hash = sha(POLICY)
    write_new(MANIFEST, {
        "schema": "strict_r3_smooth_capital_protection_policy_seal_v1",
        "base_policy": str(BASE_POLICY.relative_to(ROOT)),
        "base_policy_sha256": sha(BASE_POLICY),
        "policy": str(POLICY.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "frozen_block": {
            "smooth_capital_protection_enabled": True,
            "protection_unit": "raw_decision_time_atr",
            "protection_activation_atr": 1.5,
            "protection_strength": 0.5,
            "protection_power": 1.5,
        },
        "legacy_capital_protection_disabled": True,
        "adaptive_exit_role": "trailing_activation_modulator_only",
        "same_bar_arm_and_trigger_prohibited": True,
        "long_stop_amendment": "upward_only_reduce_only",
    })
    execution = copy.deepcopy(json.loads(BASE_EXECUTION.read_text()))
    runtime_paths = list(dict(execution["runtime_code_sha256"]).keys())
    execution.update({
        "version_note": (
            "v69: sealed smooth capital-protection successor. Entry-time ATR "
            "is immutable; the one-minute monitor uses hard stop -> prior smooth "
            "lock -> trailing -> fast adverse -> next-bar updates."
        ),
        "exit_policy": str(POLICY.relative_to(ROOT)),
        "exit_policy_sha256": policy_hash,
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "position_monitor": (
            "persistent one-minute controller: raw decision-time signal ATR, "
            "prior-bar-only smooth lock, upward-only reduce-only stop amendments; "
            "Adaptive Exit V1 modulates trailing activation only."
        ),
        "runtime_code_sha256": {
            item: sha(ROOT / item) for item in runtime_paths
        },
    })
    authorization = copy.deepcopy(json.loads(BASE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-17 to promote the sealed smooth "
            "capital-protection parent-policy successor. Model/admission/portfolio "
            "semantics are unchanged."
        ),
        "exit_policy": str(POLICY.relative_to(ROOT)),
        "exit_policy_sha256": policy_hash,
        "preserved_gates": list(authorization.get("preserved_gates") or []) + [
            "raw_decision_time_ATR_smooth_protection",
            "same_bar_smooth_arm_and_trigger_prohibited",
            "upward_only_reduce_only_stop_amendments",
        ],
    })
    write_new(AUTHORIZATION, authorization)
    execution["activation_authorization_sha256"] = sha(AUTHORIZATION)
    write_new(EXECUTION, execution)
    print(json.dumps({
        "policy": str(POLICY), "policy_sha256": policy_hash,
        "execution": str(EXECUTION), "execution_sha256": sha(EXECUTION),
        "authorization": str(AUTHORIZATION), "authorization_sha256": sha(AUTHORIZATION),
        "manifest": str(MANIFEST), "manifest_sha256": sha(MANIFEST),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
