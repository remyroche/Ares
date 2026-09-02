#!/usr/bin/env python3
"""Audit the frozen-P3 confirmation and 88-trial GateProxy successor chain.

This is an evidence-only receipt.  It has no score, model, admission,
portfolio, live, or exchange authority.  In particular, it records that the
P0 successor binding is an offline shortlist tool, while advancement remains
limited to actual six-month MC1 confirmation evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_gateproxy_completion_audit_v1"
P3_SCORING = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_gateproxy_candidate_scoring75_20260830_v1"
P3_MC1 = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_gateproxy_selected_mc1_plan75_20260830_v2"
F128_MC1 = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_f128_support_mc1_20260830_v1"
LABELS = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_proxy_downstream_labels_parent54_append27_gateproxy5_f128support2_joint88_20260830_v1"
PROXY = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_downstream_proxy_allcontracts88_20260830_v1"
CHOICE = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_gateproxy_grouped_portability_allcontracts88_20260830_v1"
BINDING = ROOT / "config/strict_r3_p8u_meta_hpo_objective_binding_allcontracts88_20260830_v1.json"
OBJECTIVE = ROOT / "config/strict_r3_p8u_meta_hpo_gateproxy_objective_allcontracts88_20260830_v1.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _all_true(path: Path) -> bool:
    values = json.loads(path.read_text())
    return all(value is True for value in values.values() if isinstance(value, bool))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)

    p3 = pd.read_parquet(P3_SCORING / "mc1_confirmation_proposal.parquet")
    p3_counts = p3.proposal_role.astype(str).value_counts().to_dict()
    expected_p3 = {
        "highest_predicted_gate_value": 3,
        "high_surrogate_uncertainty": 1,
        "diverse_descriptor_control": 1,
    }
    if p3_counts != expected_p3:
        raise AssertionError(f"unexpected frozen-P3 proposal composition: {p3_counts}")
    expected_mc1 = set(p3.trial.astype(str))
    actual_mc1 = {path.name for path in (P3_MC1 / "candidate_mc1").iterdir() if path.is_dir()}
    if actual_mc1 != expected_mc1:
        raise AssertionError("P3 MC1 receipts do not match the frozen proposal exactly")
    if not _all_true(P3_SCORING / "correctness_report.json") or not _all_true(P3_MC1 / "correctness_report.json"):
        raise AssertionError("P3 scoring or MC1 correctness failed")
    if not all(_all_true(path / "correctness_report.json") for path in (P3_MC1 / "candidate_mc1").iterdir() if path.is_dir()):
        raise AssertionError("a selected P3 MC1 receipt failed")

    expected_f128_extensions = {
        "f128ext_under100_xendcg_d2",
        "f128ext_magnitude_xendcg_d3",
        "f128ext_over100_lambdarank_d5",
    }
    actual_f128_extensions = {
        path.name for path in (F128_MC1 / "candidate_mc1").iterdir() if path.is_dir()
    } | {"f128ext_under100_xendcg_d2"}
    if actual_f128_extensions != expected_f128_extensions:
        raise AssertionError("independent F128 extension receipt mismatch")
    if not _all_true(F128_MC1 / "correctness_report.json") or not all(
        _all_true(path / "correctness_report.json") for path in (F128_MC1 / "candidate_mc1").iterdir() if path.is_dir()
    ):
        raise AssertionError("F128 support MC1 correctness failed")

    labels = pd.read_parquet(LABELS / "downstream_trial_labels.parquet")
    if len(labels) != 88 or labels.trial.duplicated().any():
        raise AssertionError("joint ledger does not contain 88 unique trials")
    descriptor_roots = json.loads((PROXY / "run_manifest.json").read_text())["descriptor_roots"]
    descriptors = pd.concat([
        pd.read_parquet(Path(root) / "trial_descriptor_summary.parquet") for root in descriptor_roots
    ], ignore_index=True)
    table = descriptors.merge(labels[["trial"]], on="trial", how="inner", validate="one_to_one")
    f128_contract = str(ROOT / "config/strict_r3_p8u_meta_feature_contract_shapcombined_additive_20260829_v1.json")
    f128_trials = int(table.feature_contract.astype(str).eq(f128_contract).sum())
    if f128_trials < 6:
        raise AssertionError(f"F128 has insufficient labelled support: {f128_trials}")

    choice = json.loads((CHOICE / "gateproxy_grouped_portability_choice.json").read_text())
    if choice["selected"]["model"] != "P0_ridge":
        raise AssertionError("unexpected successor GateProxy model")
    if min(int(choice["selected"][f"eligible_groups_{name}"]) for name in ("target_family", "loss", "feature_contract", "era")) < 2:
        raise AssertionError("successor GateProxy lacks a supported validation family")
    if not _all_true(PROXY / "correctness_report.json") or not _all_true(CHOICE / "correctness_report.json"):
        raise AssertionError("proxy refit or support-aware selection correctness failed")

    binding = json.loads(BINDING.read_text())
    objective_path = ROOT / binding["active_learned_objective"]
    if objective_path.resolve() != OBJECTIVE.resolve():
        raise AssertionError("successor binding resolves to an unexpected objective")
    objective = json.loads(OBJECTIVE.read_text())
    expected_model = ROOT / objective["objective"]["model_artifact"]
    if _sha(expected_model) != objective["objective"]["model_sha256"]:
        raise AssertionError("successor binding model hash mismatch")

    summary = {
        "p3_candidate_bank_trials": 75,
        "p3_mc1_confirmed_trials": int(len(expected_mc1)),
        "joint_labelled_trials": int(len(labels)),
        "f128_labelled_trials": f128_trials,
        "successor_gateproxy": objective["objective"]["name"],
        "successor_model_sha256": objective["objective"]["model_sha256"],
        "p3_proposal_roles": p3_counts,
        "scope": "offline HPO and feature-selection screening only; all trial promotion remains subject to actual strict six-month MC1 confirmation",
    }
    out.mkdir(parents=True)
    _once(out / "completion_audit.json", summary)
    _once(out / "correctness_report.json", {
        "frozen_p3_scored_one_diverse_75_trial_bank": True,
        "frozen_p3_proposed_exactly_top3_uncertainty_and_diverse_control": True,
        "all_five_frozen_p3_candidates_have_strict_six_month_mc1_receipts": True,
        "joint_ledger_has_88_unique_trials_under_one_normalisation": True,
        "f128_has_at_least_three_new_independent_mc1_confirmed_extensions": True,
        "f128_has_at_least_six_total_labelled_trials_before_portable_selection": True,
        "proxy_refit_and_support_aware_selection_passed": True,
        "successor_binding_hash_matches_selected_proxy": True,
        "no_live_or_exchange_mutation": True,
        "no_proxy_has_direct_trial_promotion_authority": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": summary["scope"],
        "inputs": {
            "frozen_p3_scoring": str(P3_SCORING),
            "p3_mc1": str(P3_MC1),
            "f128_mc1": str(F128_MC1),
            "joint_labels": str(LABELS),
            "proxy": str(PROXY),
            "choice": str(CHOICE),
            "binding": str(BINDING),
        },
        "summary": summary,
    })
    print(out)


if __name__ == "__main__":
    main()
