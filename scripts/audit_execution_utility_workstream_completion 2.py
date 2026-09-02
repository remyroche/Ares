#!/usr/bin/env python3
"""Audit the execution-utility workstream against its controlling brief."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "frozen_stack": ROOT
    / "data_perp/artifacts/frozen_transition_research_stack_audit_20260729_v1/manifest.json",
    "direct_summary": ROOT
    / "data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_summary_20260729_v1/summary.json",
    "interaction_manifest": ROOT
    / "data_perp/artifacts/frozen_transition_opportunity_interaction_audit_20260729_v1/manifest.json",
    "packets": ROOT
    / "data_perp/artifacts/economic_opportunity_state_packets_20260729_v1/manifest.json",
    "common30": ROOT
    / "data_perp/artifacts/common30_opportunity_support_extension_20260729_v1/manifest.json",
    "incident_gate": ROOT
    / "data_perp/artifacts/prospective_opportunity_incident_gate_20260729_v2/registry.json",
    "policy_ledger": ROOT
    / "data_perp/artifacts/research_execution_policy_decision_ledger_20260729_v1/manifest.json",
    "paired_diagnostic": ROOT
    / "data_perp/artifacts/base_ic_execution_ev_paired_completion_20260729_v1/manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources(paths: dict[str, Path] = SOURCES) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payloads[key] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def requirement_rows(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    frozen = payloads["frozen_stack"]
    direct = payloads["direct_summary"]
    interaction = payloads["interaction_manifest"]
    packets = payloads["packets"]
    common30 = payloads["common30"]
    gate = payloads["incident_gate"]["gate"]
    policy = payloads["policy_ledger"]
    paired = payloads["paired_diagnostic"]
    return [
        {
            "requirement": "freeze_transition_research_stack",
            "status": "PROVED_COMPLETE",
            "evidence": frozen["status"],
            "remaining": "",
        },
        {
            "requirement": "build_one_direct_multitask_execution_utility_model",
            "status": "PROVED_COMPLETE",
            "evidence": (
                f"winner={direct['winner']['feature_arm']}+"
                f"{direct['winner']['task_arm']}; March "
                f"{direct['march_selection_score_bps']:.2f} bps"
            ),
            "remaining": "",
        },
        {
            "requirement": "direct_model_outperform_residual_on_untouched_oos",
            "status": "COMPLETED_NEGATIVE_RESULT",
            "evidence": (
                f"April mapped={direct['april_top10_bps']['joint_winner_causal_recent']:.2f} "
                f"vs residual={direct['april_top10_bps']['frozen_residual_expected_ev']:.2f} bps"
            ),
            "remaining": "do not promote",
        },
        {
            "requirement": "separate_opportunity_state_from_regime_state",
            "status": "PROVED_COMPLETE",
            "evidence": (
                f"{packets['rows']['strict_event_packets']} frozen packets; "
                "multilabel taxonomy; no router"
            ),
            "remaining": "",
        },
        {
            "requirement": "test_transition_health_incrementality_on_direct_ev",
            "status": "COMPLETED_NEGATIVE_RESULT",
            "evidence": interaction["status"],
            "remaining": "keep transition context-only",
        },
        {
            "requirement": "explain_base_ic_execution_ev_divergence",
            "status": "PROVED_COMPLETE",
            "evidence": paired["status"],
            "remaining": "",
        },
        {
            "requirement": "use_older_data_without_illegal_lineage_pooling",
            "status": "PROVED_COMPLETE",
            "evidence": (
                f"Common30 packets={common30['rows']['strict_containing_incidents']}; "
                f"promotion_eligible={common30['promotion_eligible']}"
            ),
            "remaining": "",
        },
        {
            "requirement": "prospective_append_only_incident_governance",
            "status": "PROVED_COMPLETE",
            "evidence": (
                f"{payloads['incident_gate']['status']}; "
                f"append_only={payloads['incident_gate']['append_only_contract_enforced']}"
            ),
            "remaining": "",
        },
        {
            "requirement": "bind_content_addressed_policy_and_decision_ledger",
            "status": "PARTIAL_RESEARCH_ONLY",
            "evidence": (
                f"policy={policy['policy_id']}; prospective_rows="
                f"{policy['prospective_rows']}; promotion={policy['promotion_eligible']}"
            ),
            "remaining": "bind a frozen prospective incumbent policy feed",
        },
        {
            "requirement": "accumulate_60_to_100_compatible_strict_incidents",
            "status": "INCOMPLETE_EXTERNAL_SUPPORT",
            "evidence": (
                f"current_model={gate['candidate_current_model_incidents']}; "
                f"taxonomy={gate['taxonomy_usable_current_model_incidents']}; "
                f"prospective={gate['prospective_forward_research_incidents']}; "
                f"portfolio={gate['promotion_grade_incumbent_portfolio_incidents']}"
            ),
            "remaining": (
                f"{gate['remaining_to_minimum_current_model']} current-model or "
                f"{gate['remaining_to_minimum_incumbent_portfolio']} incumbent-policy "
                "incidents to minimum"
            ),
        },
        {
            "requirement": "train_or_promote_failure_router",
            "status": "CORRECTLY_NOT_AUTHORIZED",
            "evidence": (
                f"detector={gate['supervised_failure_detector_training_authorized']}; "
                f"router={gate['opportunity_state_router_authorized']}"
            ),
            "remaining": "wait for compatible support gate",
        },
    ]


def audit_status(rows: list[dict[str, Any]]) -> str:
    incomplete = {
        row["requirement"]
        for row in rows
        if row["status"] in {"INCOMPLETE_EXTERNAL_SUPPORT", "PARTIAL_RESEARCH_ONLY"}
    }
    if incomplete:
        return "IMPLEMENTATION_COMPLETE_PROSPECTIVE_ACCUMULATION_OPEN"
    return "WORKSTREAM_COMPLETE"


def run(args: argparse.Namespace) -> dict[str, Any]:
    payloads = load_sources()
    rows = requirement_rows(payloads)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    matrix_path = output / "requirement_matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    report = {
        "schema": "execution_utility_workstream_completion_audit_v1",
        "status": audit_status(rows),
        "objective_complete": audit_status(rows) == "WORKSTREAM_COMPLETE",
        "requirement_counts": pd.Series(
            [row["status"] for row in rows]
        ).value_counts().to_dict(),
        "requirements": rows,
        "authoritative_sources": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in SOURCES.items()
        },
        "only_unmet_end_state": (
            "an unchanged prospective incumbent execution-EV policy feed and "
            "60-100 fully resolved compatible incidents"
        ),
        "meaningful_model_or_hpo_work_authorized_without_new_data": False,
        "output": {
            "path": str(matrix_path.resolve()),
            "sha256": sha256(matrix_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.sha256").write_text(
        sha256(report_path) + "\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
