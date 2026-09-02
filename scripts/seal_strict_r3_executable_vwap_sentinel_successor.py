#!/usr/bin/env python3
"""Seal the user-approved frozen-threshold executable-VWAP exit sentinel.

The successor changes only live exit execution:

* a fresh, full-size bid-side VWAP is checked every 30 seconds;
* it may act only against the threshold persisted by the prior completed
  one-minute policy bar, after the recorded entry half-spread and adverse
  entry slippage allowance; and
* the native Kraken ``last`` stop becomes a lower, 50-bps catastrophe
  backstop.  It is still required for every live position.

No model, feature, Geometry/K9, calibration, admission, portfolio, entry, or
parent policy parameter is changed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "config/strict_r3_kraken_live_execution_v95_v117_"
    "bcf_current_dual_v96_runtime_bridgefix.json"
)
OUTPUT = ROOT / (
    "config/strict_r3_kraken_live_execution_v96_v118_"
    "executable_vwap_sentinel.json"
)
LIVE_EXECUTION = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    source = json.loads(SOURCE.read_text())
    if bool(source.get("full_vwap_hard_stop_monitor_enabled", False)):
        raise ValueError("sentinel successor requires the obsolete full-VWAP hard stop off")
    if not bool(source.get("close_based_hard_stop_monitor_enabled", False)):
        raise ValueError("sentinel successor requires the sealed close-based parent stop")
    if not bool(source.get("protective_stop_exit_vwap_adjustment", False)):
        raise ValueError("sentinel successor requires an executable-VWAP native stop")

    payload = copy.deepcopy(source)
    payload["executable_vwap_frozen_threshold_sentinel_enabled"] = True
    # Reuse the already sealed maximum adverse exit allowance rather than
    # inventing another risk magnitude.  The parent policy threshold remains
    # the sentinel authority; this distance only makes the resident native
    # stop a non-primary catastrophe backstop.
    payload["native_last_stop_backstop_bps"] = float(
        payload["maximum_exit_slippage_bps"]
    )
    payload["position_monitor"] = (
        "every 30 seconds: use only the prior completed-1m frozen policy "
        "threshold; sell remaining long size through fresh Kraken bids; "
        "exit reduce-only when executable VWAP is below the entry-friction "
        "discounted threshold. Completed 1m high updates MFE/trailing/smooth "
        "state only for the next interval; native last stop remains 50-bps "
        "lower catastrophe backstop."
    )
    payload["version_note"] = (
        "v118: user-approved executable-VWAP frozen-threshold sentinel. "
        "Every 30 seconds it evaluates remaining-size bid VWAP against only "
        "the prior completed-1m parent-policy threshold after entry half-spread "
        "and adverse entry-slippage allowance. Completed-bar high updates "
        "trailing/smooth state for the next interval only. The native Kraken "
        "last stop is retained 50 bps lower as a catastrophe backstop. All "
        "models, Geometry/K9, calibration, admission, auction, entries and "
        "frozen rich-policy parameters are unchanged."
    )
    hashes = dict(payload["runtime_code_sha256"])
    hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    payload["runtime_code_sha256"] = hashes
    payload["runtime_reseal_predecessors"] = list(
        payload.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "frozen_threshold_executable_vwap_sentinel_v1",
        "predecessor_execution": str(SOURCE.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE),
        "allowed_runtime_code_paths": [LIVE_EXECUTION],
        "changed_execution_fields": [
            "executable_vwap_frozen_threshold_sentinel_enabled",
            "native_last_stop_backstop_bps",
            "position_monitor",
        ],
        "native_backstop_bps_source": "maximum_exit_slippage_bps",
    }]
    write_new(OUTPUT, payload)
    print(json.dumps({
        "execution": str(OUTPUT.relative_to(ROOT)),
        "execution_sha256": sha(OUTPUT),
        "runtime_hash": hashes[LIVE_EXECUTION],
        "native_last_stop_backstop_bps": payload["native_last_stop_backstop_bps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
