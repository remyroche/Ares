#!/usr/bin/env python3
"""Materialise immutable per-arm prehistory plans for selected Meta trials.

The learned downstream-value proxy labels a deliberately diverse subset of
strict-OOF Meta trials.  Each selected trial needs an earlier target-free score
continuation so the frozen six-month MC1 map can warm up without using later
labels.  This utility only partitions the already-selected trial configurations
by their declared target/query arm; it neither ranks nor fits a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


SCHEMA = "strict_r3_p8u_meta_proxy_prehistory_plan_v1"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, default=None,
                        help="optional exact feature-contract slice of a mixed confirmation plan")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.selection_root.resolve()
    source = root / "selected_trial_plan.json"
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(source.read_text())
    if not isinstance(plan, list) or not plan:
        raise AssertionError("selected trial plan must be non-empty")
    if args.feature_contract is not None:
        requested = str(args.feature_contract.resolve())
        plan = [record for record in plan if str(Path(str(record.get("source_feature_contract"))).resolve()) == requested]
        if not plan:
            raise AssertionError("feature-contract slice contains no selected trials")
    names = [str(record.get("trial")) for record in plan]
    if len(names) != len(set(names)):
        raise AssertionError("selected trial plan contains duplicate trials")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_contracts: set[str] = set()
    source_roots: set[str] = set()
    source_trial_indices: dict[str, int] = {}
    for record in plan:
        score_root = Path(str(record.get("source_score_root"))).resolve()
        manifest = json.loads((score_root / "run_manifest.json").read_text())
        trials = manifest.get("trials")
        if not isinstance(trials, list) or not trials:
            raise AssertionError(f"{score_root}: missing original trial order")
        for index, source_trial in enumerate(trials):
            name = str(source_trial.get("name"))
            if not name:
                raise AssertionError(f"{score_root}: trial without name")
            existing = source_trial_indices.get(name)
            if existing is not None and existing != index:
                raise AssertionError(f"{name}: conflicting original trial index")
            source_trial_indices[name] = index
    for record in plan:
        trial = record.get("trial_config")
        if not isinstance(trial, dict):
            raise AssertionError("selected trial has no complete trial configuration")
        name = str(trial.get("name"))
        arm = str(trial.get("arm_name"))
        if not name or not arm or name != str(record.get("trial")):
            raise AssertionError("selected trial lineage does not agree with trial configuration")
        if name not in source_trial_indices:
            raise AssertionError(f"{name}: absent from its source score receipt")
        # The original bank's position is part of the deterministic fold seed
        # contract.  Retaining it prevents a subset replay from silently
        # changing a selected trial's seed merely because neighbours were not
        # selected for expensive MC1 labelling.
        replay_trial = dict(trial)
        replay_trial["strict_oof_trial_index"] = int(source_trial_indices[name])
        grouped[arm].append(replay_trial)
        source_contracts.add(str(record.get("source_feature_contract")))
        source_roots.add(str(record.get("source_score_root")))
    if len(source_contracts) != 1:
        raise AssertionError("this prehistory planner supports one exact frozen feature contract only")

    out.mkdir(parents=True)
    arm_audit: list[dict[str, object]] = []
    for arm, trials in sorted(grouped.items()):
        path = out / f"{arm}.json"
        _once(path, trials)
        arm_audit.append({
            "arm": arm,
            "trials": len(trials),
            "trial_names": [str(trial["name"]) for trial in trials],
            "strict_oof_trial_indices": [int(trial["strict_oof_trial_index"]) for trial in trials],
            "sha256": _sha(path),
        })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline selected-Meta prehistory plan only; no ranking, labels, MC1, admission, portfolio, live, or exchange mutation",
        "selection_root": str(root),
        "selection_plan_sha256": _sha(source),
        "feature_contract_filter": str(args.feature_contract.resolve()) if args.feature_contract else None,
        "selected_trials": len(plan),
        "feature_contract": next(iter(source_contracts)),
        "existing_oof_score_roots": sorted(source_roots),
        "arms": arm_audit,
        "original_trial_seed_indices_preserved": True,
        "selection_authority": "none; this is a deterministic partition of an existing representative sample",
    })
    print(out)


if __name__ == "__main__":
    main()
