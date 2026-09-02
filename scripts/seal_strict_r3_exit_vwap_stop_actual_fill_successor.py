#!/usr/bin/env python3
"""Seal the actual-filled-size correction for exit-VWAP stop protection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "config/strict_r3_inference_overlay_long_20260801_v84_bcf_current_dual_recovered_terminal_1000.json"
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v87_bcf_current_dual_exit_vwap_stop_bridgefix.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v88_bcf_current_dual_exit_vwap_stop_actual_fill.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v85_v107_bcf_current_dual_exit_vwap_stop_bridgefix.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v86_v108_bcf_current_dual_exit_vwap_stop_actual_fill.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v15_bcf_current_dual_exit_vwap_stop_bridgefix.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v16_bcf_current_dual_exit_vwap_stop_actual_fill.json"
CHANGED = [
    "extreme_price_movements/inference/strict_r3_live_execution.py",
    "extreme_price_movements/inference/trade_executor.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = json.loads(SOURCE_OVERLAY.read_text())
    hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    hashes["extreme_price_movements/inference/strict_r3_live_execution.py"] = sha(
        ROOT / "extreme_price_movements/inference/strict_r3_live_execution.py"
    )
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v88: full-size exit-VWAP stop protection corrected to recompute the "
        "bid-side estimate for the actual accepted fill quantity. Frozen model, "
        "admission, portfolio and rich-policy targets remain unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": ["extreme_price_movements/inference/strict_r3_live_execution.py"],
        "reason": "Use actual protected contracts, not intended contracts, for the full-size exit VWAP stop adjustment.",
    }
    write_new(OUT_OVERLAY, overlay)

    auth = json.loads(SOURCE_AUTH.read_text())
    auth.update({
        "authorization_source": (
            "User-approved full-size exit-VWAP protective-stop policy, with "
            "the expected exit price recomputed against actual filled contracts."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, auth)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution_hashes = {
        source: sha(ROOT / source)
        for source in dict(execution["runtime_code_sha256"])
    }
    execution.update({
        "version_note": (
            "v108: full-size exit-VWAP stop adjustment recomputed on actual "
            "Kraken fill quantity; all frozen score, admission, portfolio and "
            "rich-policy semantics unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "runtime_code_sha256": execution_hashes,
        "runtime_reseal_predecessors": [{
            "current_inference_bundle_sha256": sha(OUT_OVERLAY),
            "predecessor_inference_bundle": str(PREDECESSOR.relative_to(ROOT)),
            "predecessor_inference_bundle_sha256": sha(PREDECESSOR),
            "allowed_runtime_code_paths": CHANGED,
            "added_runtime_code_paths": [],
        }],
    })
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OUT_OVERLAY.relative_to(ROOT)), "overlay_sha256": sha(OUT_OVERLAY),
        "authorization": str(OUT_AUTH.relative_to(ROOT)), "authorization_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)), "execution_sha256": sha(OUT_EXECUTION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
