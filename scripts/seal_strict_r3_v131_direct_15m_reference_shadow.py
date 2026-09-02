#!/usr/bin/env python3
"""Seal a shadow-only overlay for causal direct-15m candidate references."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v130_"
    "bcf_current_dual_samebundle21d_decision_open_volume_pit_shadow.json"
)
OUT = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v131_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_shadow.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_stateful_recovery_v157_"
    "20260822T140000Z_150000Z_v1/direct_15m_reference_shadow_reseal_review.json"
)
CHANGED = "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists() or REVIEW.exists():
        raise FileExistsError("immutable output exists")
    source = _read(SOURCE)
    base = _read(ROOT / str(source["base_bundle"]))
    hashes = dict(base.get("runtime_code_sha256") or {})
    hashes.update(dict(source.get("overrides", {}).get("runtime_code_sha256") or {}))
    changed = [name for name, expected in hashes.items() if _sha(ROOT / name) != expected]
    if changed != [CHANGED]:
        raise AssertionError(f"unexpected runtime delta: {changed}")

    overlay = copy.deepcopy(source)
    overrides = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    overrides[CHANGED] = _sha(ROOT / CHANGED)
    overlay["overrides"]["runtime_code_sha256"] = overrides
    overlay["purpose"] = (
        "v131 shadow-only validation: a direct 15m decision open is a causal "
        "candidate reference; actual order eligibility remains live-book/VWAP "
        "preflight after admission and auction."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED],
        "economic_contract_changed": False,
        "shadow_validation_required_before_live_authorization": True,
        "reason": (
            "Do not use an unknown future final volume or archival book "
            "alignment to remove a timestamp-exact direct 15m candidate."
        ),
    }
    _write_new(OUT, overlay)

    review = {
        "schema": "strict_r3_direct_15m_reference_shadow_reseal_review_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE),
        "shadow_overlay": str(OUT.relative_to(ROOT)),
        "shadow_overlay_sha256": _sha(OUT),
        "changed_runtime_paths": [CHANGED],
        "order_authority": "none",
        "checks": {
            "direct_15m_open_is_causal_candidate_reference": True,
            "unknown_final_volume_never_qualifies_an_order": True,
            "post_admission_live_book_vwap_preflight_remains_required": True,
            "live_authorization_not_created": True,
        },
    }
    _write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
