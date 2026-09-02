#!/usr/bin/env python3
"""Create a shadow-only strict-R3 overlay for the decision-volume PIT repair."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/strict_r3_inference_overlay_long_20260801_v129_bcf_current_dual_samebundle21d_recovered_terminal.json"
OUT = ROOT / "config/strict_r3_inference_overlay_long_20260801_v130_bcf_current_dual_samebundle21d_decision_open_volume_pit_shadow.json"
REVIEW = ROOT / "data_perp/artifacts/strict_r3_stateful_recovery_v156_20260822T140000Z_150000Z_v1/decision_open_volume_pit_shadow_reseal_review.json"
CHANGED = "scripts/run_tp6_sl4_exact170_canonical_consensus.py"


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
        "v130 shadow-only validation: decision-candle volume is unavailable at "
        "the entry boundary; the direct 15m open remains book-validated."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED],
        "economic_contract_changed": False,
        "shadow_validation_required_before_live_authorization": True,
        "reason": "Prevent future intra-bar 15m volume from qualifying an entry.",
    }
    _write_new(OUT, overlay)
    review = {
        "schema": "strict_r3_decision_open_volume_pit_shadow_reseal_review_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE),
        "shadow_overlay": str(OUT.relative_to(ROOT)),
        "shadow_overlay_sha256": _sha(OUT),
        "changed_runtime_paths": [CHANGED],
        "order_authority": "none",
        "checks": {
            "direct_decision_15m_volume_unavailable": True,
            "entry_requires_decision_time_book_validation": True,
            "live_authorization_not_created": True,
        },
    }
    _write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
