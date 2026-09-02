#!/usr/bin/env python3
"""Seal the approved full-size exit-VWAP protective-stop successor.

The frozen rich policy still decides the intended hard-stop *exit price*.
This successor changes only the exchange trigger: it is raised by the
observable full-position bid-side VWAP impact so the expected executable exit
equals the policy target.  It also adds the previously unbound native Kraken
stop helper to the runtime hash contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v84_"
    "bcf_current_dual_recovered_terminal_1000.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v85_"
    "bcf_current_dual_exit_vwap_stop.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v82_v104_"
    "bcf_current_dual_recovered_terminal_1000.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v83_v105_"
    "bcf_current_dual_exit_vwap_stop.json"
)
SOURCE_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v12_"
    "bcf_current_dual_recovered_terminal_1000.json"
)
OUT_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v13_"
    "bcf_current_dual_exit_vwap_stop.json"
)
STOP_RUNTIME = "extreme_price_movements/inference/strict_r3_live_execution.py"
NATIVE_STOP_RUNTIME = "extreme_price_movements/inference/trade_executor.py"


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
    runtime = dict(overlay["overrides"]["runtime_code_sha256"])
    runtime[STOP_RUNTIME] = sha(ROOT / STOP_RUNTIME)
    # The native Kraken ``stp`` payload is transitive execution logic.  It is
    # now explicitly sealed, rather than relying on an unbound helper module.
    runtime[NATIVE_STOP_RUNTIME] = sha(ROOT / NATIVE_STOP_RUNTIME)
    overlay["overrides"]["runtime_code_sha256"] = runtime
    overlay["purpose"] = (
        "v85: approved full-size exit-VWAP protective-stop successor. Frozen "
        "models, feature contract, Geometry/K9, dual admission, portfolio, "
        "rich-policy parameters and entry economics remain unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [STOP_RUNTIME, NATIVE_STOP_RUNTIME],
        "reason": (
            "Policy hard-stop is an expected full-position exit VWAP; standing "
            "Kraken native trigger compensates only currently observable sell-side impact."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = json.loads(SOURCE_AUTH.read_text())
    authorization.update({
        "authorization_source": (
            "User-approved full-size exit-VWAP protective-stop adjustment; "
            "models, admission, portfolio and frozen rich policy unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    gates = list(authorization.get("preserved_gates") or [])
    gates.append(
        "full_size_bid_book_exit_vwap_targeted_native_protective_stop"
    )
    authorization["preserved_gates"] = gates
    write_new(OUT_AUTH, authorization)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution_hashes = {
        source: sha(ROOT / source)
        for source in dict(execution["runtime_code_sha256"])
    }
    execution_hashes[NATIVE_STOP_RUNTIME] = sha(ROOT / NATIVE_STOP_RUNTIME)
    execution.update({
        "version_note": (
            "v105: full-size exit-VWAP protective stop. The rich policy stop "
            "is the intended executable VWAP; the native Kraken trigger is "
            "raised only by observable sell-side book impact."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "runtime_code_sha256": execution_hashes,
        "protective_stop_exit_vwap_adjustment": True,
        "protective_stop_book_levels": 10,
        "protective_stop": (
            "native exchange reduce-only stop-market; trigger is raised by "
            "full-position bid-side VWAP impact so expected executable exit "
            "equals frozen rich-policy stop; persist native request and book receipts"
        ),
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
        "changed_runtime_paths": [STOP_RUNTIME, NATIVE_STOP_RUNTIME],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
