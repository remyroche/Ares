#!/usr/bin/env python3
"""Materialize the authoritative Stage-B--E supersession registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
OUTPUT = ARTIFACTS / "pipeline_supersession_manifest_20260801_v1"
SCHEMA = "pipeline_supersession_manifest_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _entry(path: str, status: str, reason: str, *, allowed: list[str], blocked: list[str]) -> dict[str, Any]:
    target = ROOT / path
    return {
        "artifact": path,
        "exists": target.exists(),
        "status": status,
        "reason": reason,
        "allowed_uses": allowed,
        "blocked_uses": blocked,
        "run_manifest_sha256": sha256(target / "run_manifest.json") if (target / "run_manifest.json").exists() else None,
    }


def build() -> dict[str, Any]:
    revoked_reason = (
        "Stage-E causal audit found selected known_row_cost_bps sourced from the future-resolved "
        "label_execution_cost_return; estimated_net_if_exit_now_bps inherits the same defect."
    )
    entries = [
        _entry(
            "data_perp/artifacts/stage_d_action_features_20260731_v3",
            "REVOKED",
            revoked_reason,
            allowed=["audit", "forensic_audit", "reproducibility_audit"],
            blocked=["training", "inference", "policy_replay", "score_mapping", "promotion"],
        ),
        _entry(
            "data_perp/artifacts/stage_d_action_features_20260731_v5",
            "REVOKED",
            revoked_reason,
            allowed=["audit", "forensic_audit", "reproducibility_audit"],
            blocked=["training", "inference", "policy_replay", "score_mapping", "promotion"],
        ),
    ]
    for name in (
        "stage_d_action_mechanism_ablation_20260731_v4",
        "stage_d_action_mechanism_ablation_20260731_v5",
        "stage_d_compact_action_model_20260731_v9",
        "stage_d_compact_action_model_20260731_v10",
    ):
        entries.append(
            _entry(
                f"data_perp/artifacts/{name}",
                "REVOKED",
                revoked_reason,
                allowed=["audit", "forensic_audit", "reproducibility_audit"],
                blocked=["training", "inference", "policy_replay", "score_mapping", "promotion"],
            )
        )
    entries.append(
        _entry(
            "data_perp/artifacts/stage_e_full_candidate_overlay_20260731_v1",
            "NON_PROMOTABLE_DIAGNOSTIC",
            "Frozen overlay is arithmetic evidence only; its upstream Stage-D model is revoked and its absolute population net remains negative.",
            allowed=["audit", "forensic_audit", "reproducibility_audit"],
            blocked=["inference", "policy_replay", "score_mapping", "promotion"],
        )
    )
    reports = {}
    for name in ("STAGE_E_FINAL_REPORT.md", "STAGE_E_E2_E3_AUDIT_20260731.md"):
        path = ROOT / name
        if path.exists():
            reports[rel(path)] = sha256(path)
    return {
        "schema": SCHEMA,
        "version": "2026-08-01",
        "status": "STAGE_D_REVOKED_OPERATIONAL_FAIL_CLOSED",
        "decision_ledger": {
            "stage_b": "STAGE_B_NO_EXECUTION_TARGET_ADVANCES",
            "stage_c": "CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION",
            "stage_d": "STAGE_D_PASS_REVOKED_TARGET_PROXY_OR_CAUSAL_DEFECT",
            "portfolio": "OUT_OF_SCOPE_AND_NOT_RUN",
        },
        "source_reports_sha256": reports,
        "entries": entries,
        "policy": {
            "revoked_outputs_remain_on_disk": True,
            "default_operational_use": "fail_closed",
            "audit_use_requires_explicit_purpose": True,
            "portfolio_constraints_in_scope": False,
        },
    }


def main() -> None:
    payload = build()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest = OUTPUT / "supersession_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.sha256").write_text(f"{sha256(manifest)}  supersession_manifest.json\n", encoding="utf-8")
    (OUTPUT / "supersession_report.md").write_text(
        "# Pipeline supersession registry\n\n"
        "Stage-D outputs are retained for audit but are operationally revoked by the Stage-E causal audit.\n"
        "Portfolio constraints are explicitly out of scope.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
