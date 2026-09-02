#!/usr/bin/env python3
"""Seal a no-order only paired-current-peer recovery candidate.

The original homogeneous-28 August upstream serialization was deleted.  This
builds a *challenger-only* full schema-v6 bundle using the surviving, paired
August current-spread upstream and conversion models.  It must never be
described as byte restoration of the deleted production bundle.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/strict_r3_inference_bundle_long_20260801_v59_close_reporting_threshold_lineage.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_v165_v194_historical_dual_challenger_recovery_candidate.json"
PEER = ROOT / "data_perp/artifacts/strict_r3_lockstep_successor28_long_aug1_7_current_spread_20260812_v1/bundles/cutoff=20260801"
OUT = ROOT / "config/strict_r3_inference_bundle_long_v168_v197_paired_current_peer_historical_recovery_candidate.json"
RECEIPT = ROOT / "data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/paired_current_peer_recovery_candidate_v197/run_manifest.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError("immutable v195 output already exists")
    base = load(BASE)
    overlay = load(OVERLAY)
    if overlay.get("base_bundle") != str(BASE.relative_to(ROOT)):
        raise ValueError("unexpected recovery overlay base")
    payload = copy.deepcopy(base)
    for key, value in (overlay.get("overrides") or {}).items():
        if key in {"runtime", "paths", "sha256", "runtime_code_sha256"}:
            merged = dict(payload.get(key) or {})
            merged.update(dict(value or {}))
            payload[key] = merged
        else:
            payload[key] = copy.deepcopy(value)
    upstream = PEER / "upstream"
    conversion = PEER / "conversion"
    required = [
        upstream / "monthly_upstream_bundle.joblib", upstream / "run_manifest.json",
        conversion / "four_week_conversion_bundle.joblib", conversion / "run_manifest.json",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    paths = payload["paths"]
    hashes = payload["sha256"]
    paths["upstream_bundle_dir"] = str(upstream.relative_to(ROOT))
    paths["conversion_bundle_dir"] = str(conversion.relative_to(ROOT))
    hashes["upstream_bundle"] = sha(upstream / "monthly_upstream_bundle.joblib")
    hashes["upstream_manifest"] = sha(upstream / "run_manifest.json")
    hashes["conversion_bundle"] = sha(conversion / "four_week_conversion_bundle.joblib")
    hashes["conversion_manifest"] = sha(conversion / "run_manifest.json")
    upstream_manifest = load(upstream / "run_manifest.json")
    conversion_manifest = load(conversion / "run_manifest.json")
    producer = dict(payload.get("producer") or {})
    producer["upstream_bundle_sha256"] = upstream_manifest["bundle_sha256"]
    producer["conversion_bundle_sha256"] = conversion_manifest["bundle_sha256"]
    payload["producer"] = producer
    for relative in list((payload.get("runtime_code_sha256") or {}).keys()):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload["runtime_code_sha256"][relative] = sha(path)
    # The historical recovery is deliberately no-order and must remain
    # auditable even while unrelated working-tree code is edited.  Artifacts,
    # feature/Geometry contracts and the candidate bundle remain hard-sealed;
    # any runtime drift is surfaced in every recovery receipt and cannot be
    # used by a live bundle.
    payload["runtime"]["allow_runtime_code_drift_for_no_order_recovery"] = True
    payload["purpose"] = (
        "v197 no-order historical recovery challenger. It pairs the documented current-spread "
        "August upstream and conversion artifacts after the homogeneous-28 upstream bytes were "
        "deleted. It preserves frozen Geometry/K9 and the separately cut-off BCF challenger, but "
        "is not byte-identical to the deleted canonical current-v5 producer and has no live authority."
    )
    payload["recovery_challenger"] = {
        "schema": "strict_r3_paired_current_peer_recovery_challenger_v1",
        "canonical_byte_parity": False,
        "deleted_original_upstream_sha256": "8d8139b166dc0af69815247e2abab1999a6670b0d7cb5552a485dd7ea0006a0e",
        "paired_upstream_sha256": hashes["upstream_bundle"],
        "paired_conversion_sha256": hashes["conversion_bundle"],
        "geometry_refit": False,
        "order_submission": False,
        "live_authority": "blocked_pending_recovery_and_independent_self-parity; promotion requires separate evidence",
    }
    write_new(OUT, payload)
    receipt = {
        "schema": "strict_r3_paired_current_peer_recovery_challenger_v1",
        "status": "candidate_only_no_order",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "inference_bundle": str(OUT.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT),
        "paired_upstream": str(upstream.relative_to(ROOT)),
        "paired_upstream_sha256": hashes["upstream_bundle"],
        "paired_conversion": str(conversion.relative_to(ROOT)),
        "paired_conversion_sha256": hashes["conversion_bundle"],
        "deleted_original_upstream_sha256": "8d8139b166dc0af69815247e2abab1999a6670b0d7cb5552a485dd7ea0006a0e",
        "canonical_byte_parity": False,
        "frozen_geometry_refit": False,
        "exchange_calls": 0,
        "order_submission": False,
    }
    write_new(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
