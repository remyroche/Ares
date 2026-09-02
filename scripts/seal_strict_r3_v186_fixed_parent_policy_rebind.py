#!/usr/bin/env python3
"""Bind the preserved exact-1m parent policy into a test-only overlay.

The missing historical winner serialization is the parent expected by the
frozen Adaptive Exit V1 bundle.  An intact ``live_parent_compatible`` winner
retains the exact same SL/activation/giveback geometry.  This does not replace
the separately sealed rich live exit policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_v156_v185_"
    "geometry_universe_content_rebind.json"
)
POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_policy_hpo_live_parent_long_"
    "20260817_v2/winner.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_v158_v187_"
    "geometry_universe_exact1m_parent_policy_rebind.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_runtime_reseal_v187_"
    "exact1m_parent_policy_rebind_20260823_v1/receipt.json"
)
LOST_POLICY_SHA256 = "2dc9a145766ae383a4ab7c33e8a9f9e358175597e05582300ff0a05732673603"
EXPECTED_WINNER = {
    "sl_mult": 4.15200064332387,
    "trailing_activation_mult": 2.326224919759605,
    "fixed_trailing_gap_mult": 0.10237198997143725,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if not SOURCE_OVERLAY.is_file() or not POLICY.is_file():
        raise FileNotFoundError("source overlay or frozen policy missing")
    if OUT_OVERLAY.exists() or OUT_RECEIPT.exists():
        raise FileExistsError("v186 outputs are immutable")
    policy = _read(POLICY)
    winner = policy.get("winner") or {}
    if (
        policy.get("schema") != "strict_r3_exact_1m_policy_hpo_v1"
        or any(float(winner.get(key, float("nan"))) != value for key, value in EXPECTED_WINNER.items())
        or policy.get("live_parent_compatible") is not True
        or float((policy.get("contract") or {}).get("policy_cost_bps_once", float("nan"))) != 100.0
        or int((policy.get("contract") or {}).get("horizon_minutes", -1)) != 720
    ):
        raise ValueError("preserved policy is not the canonical fixed H12 parent")
    overlay = copy.deepcopy(_read(SOURCE_OVERLAY))
    overrides = overlay.setdefault("overrides", {})
    paths = dict(overrides.get("paths") or {})
    hashes = dict(overrides.get("sha256") or {})
    paths["exit_policy"] = str(POLICY.relative_to(ROOT))
    hashes["exit_policy"] = _sha(POLICY)
    overrides["paths"] = paths
    overrides["sha256"] = hashes
    overlay["purpose"] = (
        "v158: bind the preserved exact-1m parent-policy contract in place of "
        "a missing equivalent historical winner serialization. The rich live exit "
        "policy remains separately bound by the execution contract."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_artifact_rebind_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_artifact_paths": ["exit_policy"],
        "economic_contract_changed": False,
        "original_missing_policy_sha256": LOST_POLICY_SHA256,
        "exact1m_parent_policy": {
            **EXPECTED_WINNER,
            "horizon_minutes": 720,
            "policy_cost_bps_once": 100.0,
        },
        "reason": (
            "The original winner file is unavailable, but the preserved exact-1m "
            "live-parent winner matches the Adaptive Exit V1 bundle's frozen policy "
            "geometry exactly. This rebind does not alter rich live exits."
        ),
    }
    _write_new(OUT_OVERLAY, overlay)
    receipt = {
        "schema": "strict_r3_exact1m_parent_policy_content_rebind_v1",
        "status": "pass",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": _sha(OUT_OVERLAY),
        "original_missing_policy_sha256": LOST_POLICY_SHA256,
        "replacement_policy": str(POLICY.relative_to(ROOT)),
        "replacement_policy_sha256": _sha(POLICY),
        "exact1m_parent_policy": {
            **EXPECTED_WINNER,
            "horizon_minutes": 720,
            "policy_cost_bps_once": 100.0,
        },
        "live_rich_exit_replaced": False,
        "execution_authority_granted": False,
    }
    _write_new(OUT_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
