#!/usr/bin/env python3
"""Create a deterministic untouched frozen-parent proxy-validation set.

The expensive initial downstream-label sample is deliberately held separate
from all remaining frozen-parent Meta trials.  This utility materialises the
complete remainder without consulting proxy predictions, direct diagnostics,
or downstream outcomes.  It is therefore a true trial-level validation set
for the learned HPO objective rather than an active-learning selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "strict_r3_p8u_meta_proxy_parent_holdout_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_roots(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        arm, sep, raw = value.partition("::")
        if not sep or not arm or not raw or arm in output:
            raise ValueError("--source-score-root must be ARM::ROOT, once per arm")
        output[arm] = Path(raw).resolve()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-bank", type=Path, required=True)
    parser.add_argument("--training-selection-root", type=Path, required=True)
    parser.add_argument("--source-score-root", action="append", required=True)
    parser.add_argument("--expected-trials", type=int, default=14)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    source_roots = _parse_roots(list(args.source_score_root))
    selected_path = args.training_selection_root.resolve() / "selected_trial_plan.json"
    selected = json.loads(selected_path.read_text())
    if not isinstance(selected, list) or not selected:
        raise AssertionError("training selected-trial plan is absent or empty")
    selected_names = {str(row.get("trial")) for row in selected}
    source_feature_contracts = {str(row.get("source_feature_contract")) for row in selected}
    if len(source_feature_contracts) != 1:
        raise AssertionError("training set does not have one frozen feature contract")
    raw_bank = json.loads(args.trial_bank.resolve().read_text())
    bank = raw_bank.get("trials") if isinstance(raw_bank, dict) else raw_bank
    if not isinstance(bank, list):
        raise AssertionError("trial bank has no trial list")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for trial in bank:
        if not isinstance(trial, dict):
            continue
        name = str(trial.get("name"))
        family = str(trial.get("additive_feature_family"))
        if family != "current_frozen" or name in selected_names:
            continue
        arm = str(trial.get("arm_name"))
        root = source_roots.get(arm)
        if root is None:
            raise AssertionError(f"{name}: no source score root for arm {arm}")
        manifest = json.loads((root / "run_manifest.json").read_text())
        source_trials = manifest.get("trials")
        if not isinstance(source_trials, list) or not any(str(item.get("name")) == name for item in source_trials):
            raise AssertionError(f"{name}: absent from target-free source score receipt")
        if name in seen:
            raise AssertionError(f"duplicate bank trial {name}")
        seen.add(name)
        records.append({
            "trial": name,
            "trial_config": trial,
            "source_score_root": str(root),
            "source_feature_contract": next(iter(source_feature_contracts)),
            "selection_reason": "predeclared_remaining_frozen_parent_holdout",
            "diagnostic_stratum": "heldout_unseen_trial",
        })
    records.sort(key=lambda row: str(row["trial"]))
    if len(records) != args.expected_trials:
        raise AssertionError(f"expected {args.expected_trials} held-out frozen-parent trials, found {len(records)}")
    out.mkdir(parents=True)
    _once(out / "selected_trial_plan.json", records)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline held-out trial plan only; no proxy prediction, direct metric, MC1, admission, portfolio, or live mutation",
        "trial_bank": str(args.trial_bank.resolve()),
        "trial_bank_sha256": _sha(args.trial_bank.resolve()),
        "training_selection_root": str(args.training_selection_root.resolve()),
        "training_selection_sha256": _sha(selected_path),
        "source_score_roots": {arm: str(root) for arm, root in sorted(source_roots.items())},
        "heldout_trials": len(records),
        "feature_family": "current_frozen",
        "selection_authority": "none; complete predeclared remainder used only for out-of-sample learned-proxy falsification",
    })
    _once(out / "correctness_report.json", {
        "heldout_trials_are_disjoint_from_proxy_training_trials": not bool(seen.intersection(selected_names)),
        "all_heldout_trials_are_frozen_parent_contract": True,
        "all_heldout_trials_exist_in_target_free_score_receipts": True,
        "no_proxy_or_outcome_based_trial_selection": True,
    })
    print(out)


if __name__ == "__main__":
    main()
