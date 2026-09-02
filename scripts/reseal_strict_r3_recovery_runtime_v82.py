#!/usr/bin/env python3
"""Reseal the v60 recovery runtime after a reviewed producer-only bridge.

The sealed v81 recovery overlay predates the live producer's ability to
consume a *completed* stateful-recovery chain.  This utility creates a new
overlay and matching live contracts after proving that the only runtime source
change is that predecessor-validation bridge.  It never changes a model,
feature contract, Geometry/K9 state, admission rule, portfolio rule, or exit
policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v81_bcf_current_dual_v60_1900_state_reseal.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v82_bcf_current_dual_v60_stateful_recovery_bridge.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v79_v101_bcf_current_dual_v60_1900_state_reseal.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v80_v102_bcf_current_dual_v60_stateful_recovery_bridge.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v09_bcf_current_dual_v60_1900_state_reseal.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v10_bcf_current_dual_v60_stateful_recovery_bridge.json"
ALLOWED_CHANGED = {"scripts/run_strict_r3_live_hourly_entry_producer.py"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = json.loads(SOURCE_OVERLAY.read_text())
    runtime_hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    current = {}
    changed = set()
    for relative, expected in runtime_hashes.items():
        observed = sha(ROOT / relative)
        current[relative] = observed
        if observed != expected:
            changed.add(relative)
    if changed != ALLOWED_CHANGED:
        raise AssertionError(
            f"unexpected runtime source delta: {sorted(changed)!r}; "
            f"expected only {sorted(ALLOWED_CHANGED)!r}"
        )
    overlay["overrides"]["runtime_code_sha256"] = current
    overlay["purpose"] = (
        "v82: reviewed producer-only bridge to a fully completed v60-derived "
        "stateful recovery chain. Frozen models, feature contract, Geometry/K9, "
        "dual admission, portfolio policy and rich exit remain unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": sorted(changed),
        "reason": "Permit only a completed, zero-order stateful recovery chain as the live predecessor.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = json.loads(SOURCE_AUTH.read_text())
    authorization.update({
        "authorization_source": "User-approved v60 state recovery with a reviewed producer-only lineage bridge; no strategy change.",
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, authorization)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution.update({
        "version_note": "v102: completed v60-derived stateful recovery bridge; strategy unchanged.",
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        # A completed v82 recovery becomes an exact-v82 predecessor.  No
        # historical v60/v81 bundle may be accepted by the live producer.
        "runtime_reseal_predecessors": [],
    })
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "overlay_sha256": sha(OUT_OVERLAY),
        "authorization": str(OUT_AUTH.relative_to(ROOT)),
        "authorization_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
        "changed_runtime_paths": sorted(changed),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
