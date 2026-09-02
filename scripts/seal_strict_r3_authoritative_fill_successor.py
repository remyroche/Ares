#!/usr/bin/env python3
"""Seal the Kraken-Futures authoritative-fill reporting runtime successor.

This successor is deliberately narrow: it binds the repaired shared order
enrichment code while proving that the inference, admission and parent-policy
contracts remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v108_"
    "bcf_current_dual_full_runtime_review.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v109_"
    "bcf_current_dual_authoritative_kraken_fills.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v36_"
    "v108_full_runtime_review.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v37_"
    "v109_authoritative_kraken_fills.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v109_v133_"
    "full_runtime_review.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v110_v134_"
    "authoritative_kraken_fills.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_authoritative_kraken_fills_reseal_20260820_v1/"
    "runtime_review.json"
)
EXPECTED_RUNTIME_DELTA = {
    "extreme_price_movements/inference/trade_executor.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base_path = (ROOT / str(overlay["base_bundle"])).resolve()
    if ROOT not in base_path.parents:
        raise ValueError("base bundle escapes repository root")
    base = json.loads(base_path.read_text())
    expected = dict(base.get("runtime_code_sha256") or {})
    expected.update(
        dict((overlay.get("overrides") or {}).get("runtime_code_sha256") or {})
    )
    if not expected:
        raise ValueError("source contract has no runtime hashes")
    return expected


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    expected = resolved_runtime_hashes(source_overlay)
    actual: dict[str, str] = {}
    for relative in sorted(expected):
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise FileNotFoundError(f"sealed runtime source is unavailable: {relative}")
        actual[relative] = sha(path)
    changed = {key for key in expected if expected[key] != actual[key]}
    if changed != EXPECTED_RUNTIME_DELTA:
        raise ValueError(
            "unexpected runtime delta; expected only authoritative-fill repair, got "
            f"{sorted(changed)}"
        )

    static_before = copy.deepcopy(source_overlay)
    static_before.pop("purpose", None)
    static_before.pop("runtime_reseal", None)
    static_before.get("overrides", {}).pop("runtime_code_sha256", None)

    overlay = copy.deepcopy(source_overlay)
    overlay["overrides"]["runtime_code_sha256"][
        "extreme_price_movements/inference/trade_executor.py"
    ] = actual["extreme_price_movements/inference/trade_executor.py"]
    overlay["purpose"] = (
        "v109: Kraken Futures authoritative private-fill reporting repair. "
        "Private /fills VWAP overrides the IOC limitPrice field for realised "
        "entry/exit reporting. Frozen models, features, Geometry/K9, EV map, "
        "admission, portfolio and rich exit-policy artifacts are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": sorted(changed),
        "reason": (
            "Kraken Futures closed-order IOC limitPrice can differ from the "
            "actual private /fills execution VWAP."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    static_after = copy.deepcopy(overlay)
    static_after.pop("purpose", None)
    static_after.pop("runtime_reseal", None)
    static_after.get("overrides", {}).pop("runtime_code_sha256", None)
    if static_before != static_after:
        raise AssertionError("authoritative-fill reseal changed a non-runtime contract field")

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-20 runtime-only Kraken Futures "
            "authoritative-fill reporting repair. No model, feature, EV-map, "
            "admission, auction or exit-policy parameter changed."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": (
            "v134: runtime-only authoritative Kraken private-fill reporting "
            "repair. No frozen model, feature, EV-map, admission, portfolio, "
            "entry or parent-policy parameter change."
        ),
    })
    runtime_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative in runtime_hashes:
        runtime_hashes[relative] = sha(ROOT / relative)
    execution["runtime_code_sha256"] = runtime_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "authoritative_kraken_private_fill_reporting_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": sorted(changed),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_authoritative_kraken_fills_runtime_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "changed_runtime_paths": sorted(changed),
        "expected_runtime_hashes": {key: expected[key] for key in sorted(changed)},
        "actual_runtime_hashes": {key: actual[key] for key in sorted(changed)},
        "non_runtime_contract_exact": static_before == static_after,
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
