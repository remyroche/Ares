#!/usr/bin/env python3
"""Seal the v121 runtime-only directional-exit successor.

The active execution contract remains long-only.  This successor does not
change models, admission, auction, policy parameters, or long-side economics;
it hash-binds the reviewed implementation that handles the exit primitives
directionally so a future side-local contract cannot silently reuse long-side
book, stop, or MFE semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "config/strict_r3_kraken_live_execution_v97_v120_"
    "executable_vwap_sentinel_bridgefix.json"
)
OUTPUT = ROOT / (
    "config/strict_r3_kraken_live_execution_v97_v121_"
    "side_specific_executable_vwap_sentinel.json"
)
LIVE_EXECUTION = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable successor already exists: {OUTPUT}")
    payload = copy.deepcopy(json.loads(SOURCE.read_text()))
    runtime_hashes = dict(payload.get("runtime_code_sha256") or {})
    runtime_hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    payload["runtime_code_sha256"] = runtime_hashes
    payload["position_monitor"] = (
        "every 30 seconds: evaluate the prior completed-1m frozen policy "
        "threshold against remaining-size fresh directional executable VWAP "
        "(long sell-through-bids; short buy-through-asks), after recorded "
        "entry half-spread and directional adverse entry-slippage allowance; "
        "the completed-bar favourable extreme (high for long, low for short) "
        "updates MFE/trailing/smooth state only for the next interval; the "
        "50-bps directionally farther native last stop is the catastrophe "
        "backstop. Current production authority remains Kraken Futures "
        "long-only."
    )
    payload["version_note"] = (
        "v121: runtime-only side-specific executable-VWAP sentinel repair. "
        "The active contract remains long-only; long behaviour, models, "
        "feature/Geometry state, calibration, admission, auction, entry "
        "economics and rich-policy parameters are unchanged. The runtime now "
        "makes VWAP liquidation, threshold discount, MFE/MAE, protection/"
        "trailing tightening, native backstop and reduce-only order direction "
        "explicitly side-local for future side-enabled contracts."
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "execution": str(OUTPUT.relative_to(ROOT)),
        "execution_sha256": sha(OUTPUT),
        "runtime_sha256": runtime_hashes[LIVE_EXECUTION],
        "source_execution_sha256": sha(SOURCE),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
