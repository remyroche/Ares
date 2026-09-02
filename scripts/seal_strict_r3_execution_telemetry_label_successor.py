#!/usr/bin/env python3
"""Reseal the dual-admission successor after a telemetry-only label repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v89_bcf_current_dual_admission_only_execution.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v90_bcf_current_dual_admission_only_execution_telemetry.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v87_v109_bcf_current_dual_admission_only_execution.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v88_v110_bcf_current_dual_admission_only_execution_telemetry.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v17_bcf_current_dual_admission_only_execution.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v18_bcf_current_dual_admission_only_execution_telemetry.json"
CHANGED_RUNTIME = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = json.loads(SOURCE_OVERLAY.read_text())
    hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    hashes[CHANGED_RUNTIME] = sha(ROOT / CHANGED_RUNTIME)
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v90: telemetry-only correction. Execution receipts distinguish the "
        "model net EV from the restored raw-gross convention; frozen models, "
        "dual admission, common portfolio auction, and rich exit are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED_RUNTIME],
        "reason": "Correct only the persisted expected-EV telemetry label.",
    }
    write_new(OUT_OVERLAY, overlay)

    auth = json.loads(SOURCE_AUTH.read_text())
    auth.update({
        "authorization_source": (
            "User-approved dual-MC1 admission and common BCF-MC1 auction; "
            "v110 corrects execution expected-EV telemetry labels only."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, auth)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution.update({
        "version_note": (
            "v110: telemetry-only repair. `mapped_expected_net_bps` now "
            "reports the MC1 policy-net convention, distinct from "
            "`raw_expected_gross_bps`; no model, admission, auction, book, "
            "or rich-exit semantics changed."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "runtime_code_sha256": {
            source: sha(ROOT / source)
            for source in dict(execution["runtime_code_sha256"])
        },
        "runtime_reseal_predecessors": [{
            "current_inference_bundle_sha256": sha(OUT_OVERLAY),
            "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
            "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
            "allowed_runtime_code_paths": [CHANGED_RUNTIME],
            "added_runtime_code_paths": [],
        }],
    })
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "overlay_sha256": sha(OUT_OVERLAY),
        "authorization": str(OUT_AUTH.relative_to(ROOT)),
        "authorization_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
