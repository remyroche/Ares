#!/usr/bin/env python3
"""Seal the user-approved dual-admission-only live execution successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v88_bcf_current_dual_exit_vwap_stop_actual_fill.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v89_bcf_current_dual_admission_only_execution.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v86_v108_bcf_current_dual_exit_vwap_stop_actual_fill.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v87_v109_bcf_current_dual_admission_only_execution.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v16_bcf_current_dual_exit_vwap_stop_actual_fill.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v17_bcf_current_dual_admission_only_execution.json"
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
        "v89: user-approved execution-authority alignment. The existing "
        "dual BCF/current MC1 thresholds and common BCF-MC1 portfolio auction "
        "are the sole entry-selection authority; live book values remain "
        "mandatory telemetry and protective-stop inputs, not a second veto."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED_RUNTIME],
        "reason": (
            "Preserve the sealed common portfolio-auction order and remove "
            "post-auction live book economics as an entry veto."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    auth = json.loads(SOURCE_AUTH.read_text())
    preserved = list(auth.get("preserved_gates") or [])
    removed = {
        "live_orderbook_preflight_for_every_dual_admitted_base_routed_candidate",
        "execution_adjusted_ev_at_least_50bps",
        "absolute_100bps_decision_price_move_limit",
    }
    auth["preserved_gates"] = [gate for gate in preserved if gate not in removed] + [
        "dual_BCF_and_current_MC1_expected_EV_at_least_30bps",
        "common_BCF_MC1_portfolio_auction",
        "live_orderbook_required_for_executable_sizing_and_full_size_VWAP_targeted_protective_stop",
        "valid_exchange_fill_and_immediate_reduce_only_protective_stop",
    ]
    auth.update({
        "authorization_source": (
            "User-approved live authority: BCF-native MC1 EV >=30 bps AND "
            "current-v5 MC1 EV >=30 bps, then the common BCF-MC1 portfolio "
            "auction with no post-auction book economics veto."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, auth)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution.update({
        "version_note": (
            "v109: user-approved dual-admission-only execution authority. "
            "Both MC1 >=30-bps gates plus the common BCF-MC1 portfolio auction "
            "are the sole entry-selection authority. Live book values remain "
            "telemetry and mandatory order/stop integrity inputs only."
        ),
        "execution_entry_authority": (
            "sealed_dual_mc1_admission_then_common_portfolio_auction_only"
        ),
        "execution_book_telemetry_only": True,
        "execution_adjusted_ev": (
            "raw_expected_gross_bps - (calculated microstructure friction + "
            "adverse delay gap + 10 bps); recorded for telemetry only after "
            "dual-MC1 admission and the common portfolio auction, and never "
            "used as a post-auction entry veto or reranking signal."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
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
