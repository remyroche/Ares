#!/usr/bin/env python3
"""Seal the output-preserving BCF cache/parallel scoring runtime successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v64_bcf_current_dual_mc1_microstructure_buffer10.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v65_bcf_current_dual_mc1_cached_parallel.json"
OLD_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v63_v73_bcf_current_dual_microstructure_buffer10.json"
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v64_v74_bcf_current_dual_cached_parallel.json"
OLD_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v28_bcf_current_dual_microstructure_buffer10.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v29_bcf_current_dual_cached_parallel.json"

RUNTIME_SOURCES = (
    "scripts/run_strict_r3_live_hourly_entry_producer.py",
    "scripts/run_strict_r3_shadow_cycle.py",
    "scripts/score_strict_r3_bcf_forward.py",
    "scripts/backfill_kraken_oi_funding_sidecars.py",
    "extreme_price_movements/strict_r3_canonical_v2.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(OLD_OVERLAY.read_text()))
    overlay["purpose"] = (
        "v74 output-preserving latency successor: immutable same-model BCF "
        "prior-42 cache with causal K9 state, concurrent BCF/current scoring, "
        "and partition-batched OI/funding append merges"
    )
    hashes = overlay["overrides"]["runtime_code_sha256"]
    for relative in RUNTIME_SOURCES:
        hashes[relative] = sha(ROOT / relative)
    write_new(OVERLAY, overlay)

    execution = copy.deepcopy(json.loads(OLD_EXECUTION.read_text()))
    execution.update({
        "version_note": (
            "v74: user-approved output-preserving latency optimization. BCF cache, "
            "parallel independent scoring, and batch sidecar append preserve frozen "
            "models, admission, auction, entry economics, and rich exit policy."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "runtime_code_sha256": {
            relative: sha(ROOT / relative)
            for relative in dict(execution["runtime_code_sha256"])
        },
    })
    auth = copy.deepcopy(json.loads(OLD_AUTHORIZATION.read_text()))
    auth.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-17 for output-preserving live "
            "latency work: Kraken bulk endpoints where compatible, batch OI/funding "
            "append, immutable BCF reference caching, and concurrent BCF/current scoring."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "exit_policy": execution["exit_policy"],
        "exit_policy_sha256": execution["exit_policy_sha256"],
    })
    write_new(AUTHORIZATION, auth)
    execution["activation_authorization_sha256"] = sha(AUTHORIZATION)
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY.relative_to(ROOT)), "overlay_sha256": sha(OVERLAY),
        "execution": str(EXECUTION.relative_to(ROOT)), "execution_sha256": sha(EXECUTION),
        "authorization": str(AUTHORIZATION.relative_to(ROOT)), "authorization_sha256": sha(AUTHORIZATION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
