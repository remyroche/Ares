#!/usr/bin/env python3
"""Seal the BCF MC1 calibration cold-start runtime successor.

The change is intentionally narrow: a BCF mapper with no strictly-prior
recent resolved support falls back to its frozen structural prior (zero global
shift) and reports that fact.  It does not change any frozen model, feature,
Geometry/K9, admission threshold, auction, sizing, execution or exit-policy
parameter.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v121_"
    "bcf_current_dual_liquidation_headroom_fixed5x_terminal_trade_telemetry.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v122_"
    "bcf_current_dual_bcf_mc1_structural_prior_coldstart.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v49_"
    "v121_terminal_trade_telemetry.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v50_"
    "v122_bcf_mc1_structural_prior_coldstart.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v123_v147_"
    "terminal_trade_telemetry.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v124_v148_"
    "bcf_mc1_structural_prior_coldstart.json"
)
OUT_REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_bcf_mc1_coldstart_reseal_20260822_v1/"
    "runtime_review.json"
)
CHANGED_RUNTIME = "extreme_price_movements/strict_r3_bcf_mc1_mapper.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def static_overlay(payload: dict) -> dict:
    output = copy.deepcopy(payload)
    output.pop("purpose", None)
    output.pop("runtime_reseal", None)
    output.get("overrides", {}).pop("runtime_code_sha256", None)
    return output


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    source_execution = json.loads(SOURCE_EXECUTION.read_text())
    source_hashes = dict(source_overlay.get("overrides", {}).get("runtime_code_sha256") or {})
    if CHANGED_RUNTIME not in source_hashes:
        raise ValueError("source overlay does not seal the BCF mapper")
    actual = sha(ROOT / CHANGED_RUNTIME)
    if source_hashes[CHANGED_RUNTIME] == actual:
        raise ValueError("BCF mapper has no source delta to reseal")

    overlay = copy.deepcopy(source_overlay)
    overlay["overrides"]["runtime_code_sha256"][CHANGED_RUNTIME] = actual
    overlay["purpose"] = (
        "v122: BCF MC1 calibration cold-start repair. If no strictly-prior "
        "21-day resolved BCF outcomes are available, use the frozen structural "
        "prior with a zero shift and report explicit support/source telemetry. "
        "All models, feature contracts, Geometry/K9, thresholds, auction, "
        "sizing, execution and exit-policy parameters remain unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED_RUNTIME],
        "reason": (
            "Prevent an empty strictly-prior BCF recent-outcome window from "
            "turning the global shift and every BCF expected-EV prediction into NaN."
        ),
    }
    if static_overlay(source_overlay) != static_overlay(overlay):
        raise AssertionError("cold-start successor changed non-runtime inference semantics")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized BCF MC1 calibration cold-start repair. The fixed "
            "structural prior is used only when strict recent support is absent; "
            "all admission, portfolio, execution and policy controls are unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(source_execution)
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": (
            "v148: BCF MC1 cold-start calibration repair. No model, feature, "
            "threshold, auction, sizing, execution or parent-policy change."
        ),
    })
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "bcf_mc1_structural_prior_coldstart_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "allowed_runtime_code_paths": [CHANGED_RUNTIME],
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_bcf_mc1_coldstart_runtime_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "changed_runtime_paths": [CHANGED_RUNTIME],
        "previous_runtime_hash": source_hashes[CHANGED_RUNTIME],
        "current_runtime_hash": actual,
        "non_runtime_inference_contract_exact": (
            static_overlay(source_overlay) == static_overlay(overlay)
        ),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
