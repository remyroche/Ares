#!/usr/bin/env python3
"""Seal the bridge-only successor for the BCF same-bundle replay ledger.

This successor changes neither the model nor any economic decision rule.  It
adds the runtime validator patch which recognises the already-approved,
hash-bound BCF ledger replacement while verifying every other inherited static
field exactly against the v122 parent.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v123_bcf_current_dual_bcf_mc1_samebundle21d_replay.json"
PARENT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v122_bcf_current_dual_bcf_mc1_structural_prior_coldstart.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260822_v51_v123_bcf_mc1_samebundle21d_replay.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v125_v149_bcf_mc1_samebundle21d_replay.json"
PARENT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v124_v148_bcf_mc1_structural_prior_coldstart.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v124_bcf_current_dual_bcf_mc1_samebundle21d_replay_bridge.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260822_v52_v124_bcf_mc1_samebundle21d_replay_bridge.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v126_v150_bcf_mc1_samebundle21d_replay_bridge.json"
OUT_REVIEW = ROOT / "data_perp/artifacts/strict_r3_bcf_mc1_samebundle21d_replay_bridge_reseal_20260822_v1/runtime_review.json"
LEDGER = ROOT / "data_perp/artifacts/strict_r3_bcf_same_bundle_recent_replay_ledger_20260822T090000Z_v1/bcf_mc1_recent_replay_ledger.parquet"
PRODUCER = ROOT / "scripts/run_strict_r3_live_hourly_entry_producer.py"
MAPPER = ROOT / "extreme_price_movements/strict_r3_bcf_mc1_mapper.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if not LEDGER.is_file():
        raise FileNotFoundError(LEDGER)
    source = json.loads(SOURCE_OVERLAY.read_text())
    parent = json.loads(PARENT_OVERLAY.read_text())
    overlay = copy.deepcopy(source)
    code = overlay["overrides"]["runtime_code_sha256"]
    code[str(PRODUCER.relative_to(ROOT))] = sha(PRODUCER)
    overlay["purpose"] = (
        "v124: bridge-only validation repair for the v123 hash-bound BCF same-bundle "
        "21-day replay ledger. The BCF/current models, policy, admission, and "
        "portfolio contract are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "bridge_parent": str(PARENT_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [
            str(MAPPER.relative_to(ROOT)),
            str(PRODUCER.relative_to(ROOT)),
        ],
        "approved_calibration_artifact": str(LEDGER.relative_to(ROOT)),
        "approved_calibration_artifact_sha256": sha(LEDGER),
        "reason": "Permit only the reviewed BCF replay-ledger replacement during predecessor-state validation.",
    }
    write_new(OUT_OVERLAY, overlay)

    auth = copy.deepcopy(json.loads(SOURCE_AUTH.read_text()))
    auth.update({
        "authorization_source": "User-approved causal BCF same-bundle replay ledger, with a bridge-only validator repair.",
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, auth)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "version_note": "v150: bridge-only validation repair for the causal BCF same-bundle 21-day replay ledger; economic contract unchanged.",
    })
    execution["runtime_reseal_predecessors"] = list(execution.get("runtime_reseal_predecessors") or []) + [{
        "successor_execution_semantics": "bcf_mc1_samebundle_21d_replay_bridge_validator_v1",
        "predecessor_execution": str(PARENT_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(PARENT_EXECUTION),
        "predecessor_inference_bundle": str(PARENT_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(PARENT_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "approved_calibration_artifact": str(LEDGER.relative_to(ROOT)),
        "approved_calibration_artifact_sha256": sha(LEDGER),
        "allowed_runtime_code_paths": [
            str(MAPPER.relative_to(ROOT)),
            str(PRODUCER.relative_to(ROOT)),
        ],
        "reviewed_current_runtime": True,
        "reason": "Direct v122 state bridge; only the causal BCF ledger and bridge-validator implementation differ.",
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_bcf_mc1_samebundle21d_replay_bridge_reseal_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "parent_overlay": str(PARENT_OVERLAY.relative_to(ROOT)),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTH),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "bcf_replay_ledger": str(LEDGER.relative_to(ROOT)),
        "bcf_replay_ledger_sha256": sha(LEDGER),
        "producer_sha256": sha(PRODUCER),
        "mapper_sha256": sha(MAPPER),
        "economic_contract_changed": False,
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
