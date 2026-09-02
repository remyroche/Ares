#!/usr/bin/env python3
"""Reseal strict-R3 after adding audit-only full market snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v91_"
    "bcf_current_dual_execution_adjusted30.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v92_"
    "bcf_current_dual_execution_adjusted30_full_market_snapshot.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v89_v111_"
    "bcf_current_dual_execution_adjusted30.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v90_v112_"
    "bcf_current_dual_execution_adjusted30_full_market_snapshot.json"
)
SOURCE_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v19_"
    "bcf_current_dual_execution_adjusted30.json"
)
OUT_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v20_"
    "bcf_current_dual_execution_adjusted30_full_market_snapshot.json"
)
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
        "v92: retain full public ticker and full returned order-book depth at "
        "entry preflight and after confirmed exits. This is audit-only; frozen "
        "models, dual admission, auction, +30-bps execution-EV veto, and rich "
        "exit are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED_RUNTIME],
        "reason": "Persist full-depth public market-data audit snapshots only.",
    }
    write_new(OUT_OVERLAY, overlay)

    auth = json.loads(SOURCE_AUTH.read_text())
    auth.update({
        "authorization_source": (
            "User-approved audit-only persistence of full Kraken public ticker "
            "and returned order-book depth at entry and exit; no trading "
            "authority, selection, policy, or economics changed."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, auth)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution.update({
        "version_note": (
            "v112: audit-only full public ticker and returned book-depth capture "
            "at entry preflight and after confirmed exits. All frozen entry, "
            "portfolio, execution-EV, stop, and rich-exit behavior is unchanged."
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
