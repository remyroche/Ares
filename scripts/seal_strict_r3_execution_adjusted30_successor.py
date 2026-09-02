#!/usr/bin/env python3
"""Seal only the approved post-auction +30-bps execution-EV safety gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v90_"
    "bcf_current_dual_admission_only_execution_telemetry.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v91_"
    "bcf_current_dual_execution_adjusted30.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v88_v110_"
    "bcf_current_dual_admission_only_execution_telemetry.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v89_v111_"
    "bcf_current_dual_execution_adjusted30.json"
)
SOURCE_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v18_"
    "bcf_current_dual_admission_only_execution_telemetry.json"
)
OUT_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v19_"
    "bcf_current_dual_execution_adjusted30.json"
)
CHANGED_RUNTIME = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = json.loads(SOURCE_OVERLAY.read_text())
    hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    hashes[CHANGED_RUNTIME] = sha(ROOT / CHANGED_RUNTIME)
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v91: explicit +30-bps execution-adjusted-EV safety veto after the "
        "frozen dual-MC1 admission and common BCF-MC1 auction. It cannot "
        "rerank candidates; existing protective-stop requirements remain unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED_RUNTIME],
        "reason": "Restore only the approved post-auction execution-EV >= +30 bps veto.",
    }
    write_new(OUT_OVERLAY, overlay)

    auth = json.loads(SOURCE_AUTH.read_text())
    preserved = list(auth.get("preserved_gates", []))
    gate = "execution_adjusted_EV_at_least_30bps_after_common_portfolio_auction"
    if gate not in preserved:
        preserved.append(gate)
    auth.update({
        "authorization_source": (
            "User-approved restoration of the final non-reranking "
            "execution-adjusted-EV >= +30 bps safeguard after dual BCF/current "
            "MC1 admission and the common BCF-MC1 portfolio auction."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "preserved_gates": preserved,
    })
    write_new(OUT_AUTH, auth)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution.update({
        "version_note": (
            "v111: restores only the final execution-adjusted-EV >= +30-bps "
            "veto after the frozen dual-MC1/common-auction winner is selected. "
            "It does not restore a spread or absolute price-gap veto, and it "
            "cannot rerank candidates."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "execution_book_telemetry_only": True,
        "execution_adjusted_ev_veto_enabled": True,
        "minimum_execution_adjusted_ev_bps": 30.0,
        "execution_entry_authority": (
            "sealed_dual_mc1_admission_then_common_portfolio_auction_then_"
            "execution_adjusted_ev_ge_30bps"
        ),
        "execution_adjusted_ev": (
            "raw_expected_gross_bps - adverse_delay_gap_bps - "
            "(1.2 * live_full_spread_bps + 2 * entry_vwap_impact_bps + 10 bps); "
            "the resulting executable EV must be >= +30 bps after frozen dual "
            "MC1 admission and the common portfolio auction; it is a final veto "
            "only and never a reranking signal."
        ),
        "runtime_code_sha256": {
            source: sha(ROOT / source)
            for source in dict(execution["runtime_code_sha256"])
        },
        "runtime_reseal_predecessors": [{
            "current_inference_bundle_sha256": sha(OUT_OVERLAY),
            "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
            "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
            "allowed_runtime_code_paths": [CHANGED_RUNTIME],
            "added_runtime_code_paths": [],
        }],
    })
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "overlay_sha256": sha(OUT_OVERLAY),
        "authorization": str(OUT_AUTH.relative_to(ROOT)),
        "authorization_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
