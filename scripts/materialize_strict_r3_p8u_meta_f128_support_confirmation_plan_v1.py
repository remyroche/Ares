#!/usr/bin/env python3
"""Seal independently designed F128 trials for support-only MC1 confirmation.

F128 has deliberately been excluded from portable GateProxy selection because
the original ledger has only three labelled trials.  This utility materialises
named extension trials from the predeclared extension configuration.  It does
not rank them, inspect outcomes, or confer any promotion authority: its sole
purpose is to create the immutable input accepted by the usual strict-MC1
runner so that F128 can reach the predeclared support threshold legitimately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_f128_support_confirmation_plan_v1"
REQUIRED_SCORE_CORRECTNESS = {
    "p8u_base_target_free_score_source",
    "declared_meta_features_merged_by_exact_identity",
    "no_policy_or_path_field_in_target_free_inputs",
    "all_train_labels_resolved_before_reserve",
    "held_scores_persisted_before_held_outcome_metrics",
    "no_mc1_admission_portfolio_live_or_exchange_mutation",
}


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


def _parse_sources(raw: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in raw:
        name, separator, raw_path = value.partition("::")
        if not separator or not name or not raw_path:
            raise ValueError("--source-root must be TRIAL::ROOT")
        if name in result:
            raise ValueError(f"duplicate source for {name}")
        result[name] = Path(raw_path).resolve()
    return result


def _validate_source(*, trial: str, source: Path, expected: dict[str, Any]) -> dict[str, Any]:
    correctness = json.loads((source / "correctness_report.json").read_text())
    invalid = sorted(key for key in REQUIRED_SCORE_CORRECTNESS if correctness.get(key) is not True)
    if invalid:
        raise AssertionError(f"{trial}: target-free correctness failure {invalid}")
    manifest = json.loads((source / "run_manifest.json").read_text())
    if "offline" not in str(manifest.get("scope", "")):
        raise AssertionError(f"{trial}: source is not an offline score receipt")
    feature_contract = str(manifest.get("meta_feature_contract", ""))
    if "shapcombined" not in Path(feature_contract).name:
        raise AssertionError(f"{trial}: source does not use the F128 combined contract")
    matches = [item for item in manifest.get("trials", []) if str(item.get("name")) == trial]
    if len(matches) != 1:
        raise AssertionError(f"{trial}: source trial is missing or ambiguous")
    actual = matches[0]
    for key in ("name", "source_trial", "target", "arm_name", "feature_mode", "additive_feature_family"):
        if actual.get(key) != expected.get(key):
            raise AssertionError(f"{trial}: source configuration differs for {key}")
    if str(manifest.get("arm", {}).get("name")) != str(expected["arm_name"]):
        raise AssertionError(f"{trial}: source arm identity mismatch")
    return {"feature_contract": feature_contract, "trial_config": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--trial", action="append", required=True)
    parser.add_argument("--source-root", action="append", required=True, help="TRIAL::ROOT; repeat")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path, out = args.extension_config.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    requested = list(args.trial)
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate requested F128 trial")
    entries = json.loads(config_path.read_text())
    if not isinstance(entries, list):
        raise AssertionError("extension configuration must be a trial list")
    declared = {str(item.get("name")): item for item in entries}
    if not set(requested).issubset(declared):
        raise AssertionError("requested F128 trial is not predeclared in the extension config")
    sources = _parse_sources(args.source_root)
    if set(sources) != set(requested):
        raise AssertionError("source roots must match requested F128 trials exactly")

    records: list[dict[str, Any]] = []
    for trial in requested:
        expected = declared[trial]
        source = sources[trial]
        validated = _validate_source(trial=trial, source=source, expected=expected)
        records.append({
            "trial": trial,
            "selection_reason": "predeclared_f128_portability_support_extension",
            "proposal_role": "f128_portability_support_extension",
            "gateproxy_rank": None,
            "gateproxy_score": None,
            "gateproxy_uncertainty": None,
            "source_score_root": str(source),
            "source_feature_contract": validated["feature_contract"],
            "trial_config": validated["trial_config"],
        })

    out.mkdir(parents=True)
    _once(out / "selected_trial_plan.json", records)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline predeclared F128 support-only MC1 confirmation; no ranking, promotion, live, or exchange mutation",
        "extension_config": str(config_path),
        "extension_config_sha256": _sha(config_path),
        "trials": requested,
        "source_roots": {name: str(path) for name, path in sorted(sources.items())},
        "selection_authority": "none; establishes F128 portability support only after strict MC1 labels are added to the shared ledger",
    })
    _once(out / "correctness_report.json", {
        "all_requested_trials_are_predeclared_in_the_extension_config": True,
        "all_sources_are_target_free_strict_oof_receipts": True,
        "all_sources_use_the_f128_combined_contract": True,
        "source_trial_and_arm_identity_match_exactly": True,
        "no_proxy_ranking_outcome_mc1_portfolio_live_or_exchange_input": True,
        "no_direct_promotion_authority": True,
    })
    print(out)


if __name__ == "__main__":
    main()
