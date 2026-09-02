#!/usr/bin/env python3
"""Materialise an immutable MC1-confirmation plan from a frozen GateProxy proposal.

This utility intentionally has no selection logic of its own.  It joins the
non-authoritative P3 proposal to the descriptor receipt and then recovers each
trial's exact strict-OOF score-root configuration.  The resulting plan is the
only input accepted by the selected-MC1 runner, preserving source identity,
feature-contract provenance, and the proposal role (Top-3, uncertainty, or
diverse control).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_gateproxy_confirmation_plan_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*")) if path.is_dir() else [path]
    for member in members:
        if member.is_file():
            digest.update(str(member.relative_to(path) if path.is_dir() else member.name).encode())
            with member.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _all_true(path: Path, *, required: set[str] | None = None) -> None:
    payload = json.loads(path.read_text())
    if required is not None:
        invalid = sorted(key for key in required if payload.get(key) is not True)
        if invalid:
            raise AssertionError(f"incomplete required correctness receipt: {path}: {invalid}")
        return
    if not all(value is True for value in payload.values() if isinstance(value, bool)):
        raise AssertionError(f"incomplete correctness receipt: {path}")


def _descriptor_score_roots(manifest: dict[str, object]) -> dict[str, Path]:
    """Resolve the exact screen roots from either supported descriptor receipt."""
    values = manifest.get("score_roots", manifest.get("target_query_roots"))
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise AssertionError("descriptor receipt lacks exact score-root provenance")
    roots = {Path(value).name: Path(value).resolve() for value in values}
    if len(roots) != len(values):
        raise AssertionError("descriptor receipt has duplicate score-root basenames")
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-root", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scoring, descriptors, out = args.scoring_root.resolve(), args.descriptor_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    _all_true(scoring / "correctness_report.json")
    _all_true(descriptors / "correctness_report.json")
    proposal = pd.read_parquet(scoring / "mc1_confirmation_proposal.parquet")
    summary = pd.read_parquet(descriptors / "trial_descriptor_summary.parquet")
    if proposal.empty or proposal.trial.duplicated().any() or summary.trial.duplicated().any():
        raise AssertionError("invalid GateProxy proposal or descriptor identity")
    required_roles = {"highest_predicted_gate_value", "high_surrogate_uncertainty", "diverse_descriptor_control"}
    if not set(proposal.proposal_role.astype(str)).issubset(required_roles):
        raise AssertionError("unexpected GateProxy proposal role")
    merged = proposal.merge(
        summary.loc[:, ["trial", "score_root", "feature_contract", "feature_family", "target_family", "loss"]],
        on=["trial", "feature_contract", "feature_family", "target_family", "loss"], how="left", validate="one_to_one",
    )
    if merged.score_root.isna().any():
        raise AssertionError("GateProxy proposal did not resolve to a strict descriptor source")
    descriptor_manifest = json.loads((descriptors / "run_manifest.json").read_text())
    roots = _descriptor_score_roots(descriptor_manifest)
    records: list[dict[str, object]] = []
    for row in merged.sort_values(["gateproxy_rank", "trial"], kind="stable").itertuples(index=False):
        source = roots.get(str(row.score_root))
        if source is None:
            raise AssertionError(f"unknown score root {row.score_root}")
        _all_true(source / "correctness_report.json", required={
            "p8u_base_target_free_score_source",
            "declared_meta_features_merged_by_exact_identity",
            "no_policy_or_path_field_in_target_free_inputs",
            "all_train_labels_resolved_before_reserve",
            "held_scores_persisted_before_held_outcome_metrics",
            "no_mc1_admission_portfolio_live_or_exchange_mutation",
        })
        manifest = json.loads((source / "run_manifest.json").read_text())
        if manifest.get("scope", "").find("offline") < 0:
            raise AssertionError(f"{source}: non-offline score source")
        candidates = [item for item in manifest.get("trials", []) if str(item.get("name")) == str(row.trial)]
        if len(candidates) != 1:
            raise AssertionError(f"{row.trial}: source trial configuration is missing or ambiguous")
        trial = candidates[0]
        # HPO trial manifests store their arm once at the score-root level.  Older
        # hand-authored trial snippets did not redundantly repeat ``arm_name`` on
        # every trial.  Treat that omission as inheritance from the immutable
        # score-root receipt, while still rejecting an explicit disagreement.
        source_arm = str(manifest.get("arm", {}).get("name"))
        trial_arm = str(trial.get("arm_name", source_arm))
        if not source_arm or trial_arm != source_arm:
            raise AssertionError(f"{row.trial}: score-root arm and trial arm disagree")
        if str(manifest.get("meta_feature_contract")) != str(row.feature_contract):
            raise AssertionError(f"{row.trial}: feature contract provenance mismatch")
        # Downstream MC1 runner requires an explicit arm on every selected
        # trial.  Persist the source-receipt arm after the inheritance check,
        # rather than relying on a mutable external trial config later.
        resolved_trial = {**trial, "arm_name": source_arm}
        records.append({
            "trial": str(row.trial),
            "selection_reason": f"frozen_p3_gateproxy:{row.proposal_role}:rank={int(row.gateproxy_rank)}",
            "proposal_role": str(row.proposal_role),
            "gateproxy_rank": int(row.gateproxy_rank),
            "gateproxy_score": float(row.gateproxy_score),
            "gateproxy_uncertainty": float(row.gateproxy_uncertainty),
            "source_score_root": str(source),
            "source_arm": source_arm,
            "source_feature_contract": str(row.feature_contract),
            "trial_config": resolved_trial,
        })
    out.mkdir(parents=True)
    _once(out / "selected_trial_plan.json", records)
    pd.DataFrame(records).drop(columns=["trial_config"]).to_parquet(out / "selected_trials.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline frozen-P3 MC1 confirmation plan only; no MC1, portfolio, live, or exchange mutation",
        "scoring_root": str(scoring), "scoring_root_sha256": _sha(scoring),
        "descriptor_root": str(descriptors), "descriptor_root_sha256": _sha(descriptors),
        "selected_trials": len(records),
        "roles": {role: int((proposal.proposal_role == role).sum()) for role in sorted(required_roles)},
        "selection_authority": "proposal only; full strict matched six-month MC1 remains the advancement authority",
    })
    _once(out / "correctness_report.json", {
        "proposal_and_descriptor_receipts_passed": True,
        "each_trial_resolves_to_one_exact_target_free_score_source": True,
        "feature_contract_provenance_matches_score_source": True,
        "no_new_outcome_mc1_portfolio_live_or_exchange_input": True,
        "no_direct_trial_promotion_authority": True,
    })
    print(out)


if __name__ == "__main__":
    main()
