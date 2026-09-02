#!/usr/bin/env python3
"""Build a no-order Strict-R3 v183 runtime-refresh inference candidate.

The candidate deliberately flattens one existing inference overlay onto its
schema-v6 base because inference overlays are intentionally not composable.
It preserves every non-runtime override byte-for-byte and replaces only the
runtime-code hash map with hashes of the checked-out sources.  It does not
create an execution contract or grant exchange-writing authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v153_v181_hash_stability_capacity_fallback.json"
OUT = ROOT / "config/strict_r3_inference_overlay_long_v155_v183_runtime_refresh_candidate.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merged_static(payload: dict) -> dict:
    base = json.loads((ROOT / payload["base_bundle"]).read_text())
    overrides = copy.deepcopy(payload.get("overrides") or {})
    for key, value in overrides.items():
        if key == "runtime_code_sha256":
            continue
        if key in {"runtime", "paths", "sha256"}:
            base[key] = {**dict(base.get(key) or {}), **dict(value or {})}
        else:
            base[key] = value
    return base


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"immutable output exists: {OUT}")
    source = json.loads(SOURCE.read_text())
    if source.get("schema") != "strict_r3_inference_bundle_overlay_v1":
        raise ValueError("source must be a strict-R3 inference overlay")
    base = ROOT / str(source["base_bundle"])
    base_payload = json.loads(base.read_text())
    if not str(base_payload.get("schema") or "").startswith("strict_r3_inference_bundle_v6"):
        raise ValueError("source overlay must directly reference a schema-v6 base")

    output = copy.deepcopy(source)
    hashes = dict(output.setdefault("overrides", {}).get("runtime_code_sha256") or {})
    observed: dict[str, str] = {}
    for relative in sorted(hashes):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime path missing: {relative}")
        observed[relative] = _sha(path)
    output["overrides"]["runtime_code_sha256"] = observed
    output["purpose"] = (
        "v155/v183 candidate: audited current runtime identities over the exact "
        "v153/v181 schema-v6 base and non-runtime overrides. No model, feature "
        "contract, Geometry/K9, calibration, admission, auction, sizing, source, "
        "or exit-policy parameter reference changes. Candidate is no-order only "
        "until separately reviewed and activated."
    )
    output["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_candidate_v1",
        "supersedes": str(SOURCE.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE),
        "changed_runtime_paths": sorted(
            relative for relative, value in observed.items()
            if value != hashes[relative]
        ),
        "economic_contract_changed": "unreviewed; no-order candidate only",
    }
    if _merged_static(source) != _merged_static(output):
        raise AssertionError("runtime refresh changed a non-runtime inference override")
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate": str(OUT.relative_to(ROOT)),
        "candidate_sha256": _sha(OUT),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "changed_runtime_paths": output["runtime_reseal"]["changed_runtime_paths"],
        "non_runtime_overrides_exact": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
