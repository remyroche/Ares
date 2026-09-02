#!/usr/bin/env python3
"""Seal the already-approved dual BCF/current portfolio runtime source."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v65_bcf_current_dual_mc1_cached_parallel.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v66_bcf_current_dual_mc1_cached_parallel_dualauction.json"
OLD_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v64_v74_bcf_current_dual_cached_parallel.json"
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v65_v75_bcf_current_dual_cached_parallel_dualauction.json"
OLD_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v29_bcf_current_dual_cached_parallel.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v30_bcf_current_dual_cached_parallel_dualauction.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(OLD_OVERLAY.read_text()))
    overlay["purpose"] = (
        "v75: hash-bound activation of the already-approved BCF/current dual "
        "portfolio admission: both native MC1 EVs >=30 bps, BCF-MC1 priority, "
        "and unchanged common auction constraints"
    )
    overlay["overrides"]["runtime_code_sha256"][
        "extreme_price_movements/strict_r3_shadow_portfolio.py"
    ] = sha(ROOT / "extreme_price_movements/strict_r3_shadow_portfolio.py")
    write_new(OVERLAY, overlay)

    execution = copy.deepcopy(json.loads(OLD_EXECUTION.read_text()))
    execution.update({
        "version_note": (
            "v75: resealed approved dual BCF/current portfolio runtime. It enforces "
            "both MC1 >=30 bps and BCF-MC1 priority under unchanged common portfolio "
            "constraints; models, feature contracts, entry economics and rich exits remain unchanged."
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
            "User-approved dual admission specified before this source was written: "
            "BCF MC1 EV >=30 bps AND current-v5 MC1 EV >=30 bps, common portfolio "
            "constraints, and BCF MC1 expected EV priority. This reseal activates only "
            "that predeclared runtime implementation."
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
