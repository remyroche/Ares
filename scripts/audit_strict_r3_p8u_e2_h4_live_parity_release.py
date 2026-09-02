#!/usr/bin/env python3
"""Verify every hash and safety boundary of the P8U E2/H4 successor release.

No exchange dependency exists in this verifier.  It proves that the release
candidate is internally complete while proving the active exchange-writing
gateway remains an unchanged exclusion boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify(descriptor: Mapping[str, Any], *, label: str) -> Path:
    value, expected = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(value, str) or not isinstance(expected, str):
        raise ValueError(f"{label} lacks a hash-bound path")
    path = _path(value)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} hash mismatch")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("release audit output must be immutable")
    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "strict_r3_p8u_e2_h4_live_parity_release_candidate_v1":
        raise ValueError("unexpected E2/H4 release-candidate schema")
    if payload.get("status") != "SEALED_NO_ORDER_CANDIDATE_NOT_CONNECTED_TO_EXCHANGE" or payload.get("order_submission") is not False:
        raise ValueError("release candidate unexpectedly has order authority")
    _verify(payload["research_contract"], label="research contract")
    bundle_spec = payload["current_refit_bundle"]
    manifest = _path(str(bundle_spec["path"])) / "bundle_manifest.json"
    if not manifest.is_file() or _sha256(manifest) != str(bundle_spec["manifest_sha256"]):
        raise ValueError("current prior-resolved H0/H3/H4 bundle manifest mismatch")
    bundle_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if bundle_payload.get("order_submission") is not False or bundle_payload.get("cutoff") != "2026-08-29T00:00:00+00:00":
        raise ValueError("current bundle has unexpected authority or resolved-label cutoff")
    if bundle_payload.get("training", {}).get("e2_pairs", 0) <= 0 or bundle_payload.get("training", {}).get("h4_states", 0) <= 0:
        raise ValueError("current bundle lacks prior-resolved E2/H4 training support")
    observed_runtime: dict[str, str] = {}
    for role, descriptor in payload["runtime"].items():
        path = _verify(descriptor, label=f"runtime {role}")
        observed_runtime[role] = _sha256(path)
    parity_path = _verify(payload["parity_receipt"], label="inference/replay parity receipt")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("status") != "pass_exact_target_free_inference_replay_parity":
        raise ValueError("inference/replay parity did not pass")
    if parity.get("outcome_columns_consumed") not in (None, []) or parity.get("exchange_or_order_submission_called") is not False:
        raise ValueError("parity receipt has forbidden outcome or exchange authority")
    if any(float(parity[key]) != 0.0 for key in ("e2_h0_max_abs_delta", "e2_h3_max_abs_delta", "h4_prediction_max_abs_delta")):
        raise ValueError("inference/replay numerical parity is not exact")
    active_path = _verify(payload["active_gateway_boundary"], label="active gateway boundary")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    if active.get("order_submission") is not True:
        raise ValueError("recorded active gateway is not exchange-writing")
    active_serialized = json.dumps(active, sort_keys=True)
    if "p8u_e2_h4" in active_serialized.lower():
        raise ValueError("active exchange gateway unexpectedly references successor E2/H4 runtime")
    report = {
        "schema": "strict_r3_p8u_e2_h4_live_parity_release_audit_v1",
        "status": "pass_hash_bound_no_order_release",
        "order_submission": False,
        "release_config": str(config_path),
        "release_config_sha256": _sha256(config_path),
        "bundle_manifest_sha256": _sha256(manifest),
        "runtime_hashes": observed_runtime,
        "parity_receipt_sha256": _sha256(parity_path),
        "parity_tolerance": 0.0,
        "active_gateway_boundary_sha256": _sha256(active_path),
        "active_gateway_unchanged_and_excludes_successor": True,
        "required_before_exchange_activation": payload["promotion_prerequisites"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
