#!/usr/bin/env python3
"""Create a non-activated feature-runtime parity candidate from v75."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/strict_r3_inference_overlay_long_20260801_v66_bcf_current_dual_mc1_cached_parallel_dualauction.json"
OUT = ROOT / "config/strict_r3_inference_overlay_long_20260801_v67_feature_runtime_parity_candidate.json"
SOURCES = (
    "extreme_price_movements/features.py",
    "scripts/materialize_strict_r3_forward_features.py",
    "scripts/materialize_strict_r3_forward_features_incremental_v13.py",
    "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py",
    "scripts/run_tp6_sl4_exact170_canonical_consensus.py",
    "scripts/update_strict_r3_feature_panel_state.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    payload = copy.deepcopy(json.loads(BASE.read_text()))
    hashes = payload["overrides"]["runtime_code_sha256"]
    changed: dict[str, str] = {}
    for relative in SOURCES:
        digest = sha(ROOT / relative)
        hashes[relative] = digest
        changed[relative] = digest
    payload["purpose"] = (
        "non-activated v67 candidate: current feature/materialisation source hashes; "
        "promotion requires exact phase-zero feature, prediction and admission parity "
        "against a frozen historical live receipt"
    )
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "sha256": sha(OUT), "changed": changed}, sort_keys=True))


if __name__ == "__main__":
    main()
