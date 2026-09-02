#!/usr/bin/env python3
"""Seal learned-proxy predictions for a trial-level MC1 holdout before labels.

The holdout descriptors are strict-OOF Meta diagnostics.  This utility applies
the frozen surrogate models without opening any MC1 output; the later
MC1-replay results are joined only by a separate falsification step.
"""

from __future__ import annotations

import argparse
import __main__
import json
import os
from pathlib import Path

import joblib
import pandas as pd

import fit_strict_r3_p8u_meta_downstream_proxy_v1 as proxy


SCHEMA = "strict_r3_p8u_meta_proxy_holdout_scores_v1"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, action="append", required=True)
    parser.add_argument("--holdout-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    proxy_root = args.proxy_root.resolve()
    correctness = json.loads((proxy_root / "correctness_report.json").read_text())
    required = {
        "descriptors_are_strict_oof_and_target_free_before_outcome_metrics",
        "downstream_labels_are_matched_six_month_mc1",
        "priority_and_gate_proxies_are_separate",
        "proxy_has_no_direct_live_or_model_score_authority",
    }
    if not all(correctness.get(key) is True for key in required):
        raise AssertionError("frozen learned-proxy receipt is incomplete")
    plan = json.loads((args.holdout_plan.resolve() / "selected_trial_plan.json").read_text())
    names = [str(item.get("trial")) for item in plan]
    if len(names) != len(set(names)) or not names:
        raise AssertionError("invalid held-out trial plan")
    for root in args.descriptor_root:
        descriptor_correctness = json.loads((root.resolve() / "correctness_report.json").read_text())
        if not all(value is True for value in descriptor_correctness.values()):
            raise AssertionError(f"{root}: descriptor receipt is not clean")
    summary, _fold = proxy._read_descriptors(args.descriptor_root)
    score = summary.loc[summary.trial.isin(names)].copy()
    if len(score) != len(names) or score.trial.duplicated().any():
        raise AssertionError("held-out trial descriptor identities mismatch")
    # The initial fitter was executed as a script, so its pairwise dataclass
    # was pickled under ``__main__``.  Re-exporting the same implementation
    # restores it without refitting or changing any frozen parameters.
    setattr(__main__, "PairwiseSurrogate", proxy.PairwiseSurrogate)
    bundle = joblib.load(proxy_root / "proxy_models.joblib")
    fields = list(bundle["fields"])
    output = score.loc[:, ["trial", "target_family", "loss", "feature_family", "feature_contract"]].copy()
    for target in proxy.TARGETS:
        stem = "priority" if target.startswith("dpriority") else "gate"
        for name in ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise"):
            output[f"proxy_{stem}_{name}"] = proxy._predict(bundle["models"][f"{target}::{name}"], score[fields])
        output[f"proxy_{stem}_mean"] = output[[f"proxy_{stem}_{name}" for name in ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise")]].mean(axis=1)
    output = output.sort_values("trial", kind="stable").reset_index(drop=True)
    out.mkdir(parents=True)
    output.to_parquet(out / "holdout_proxy_predictions.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline learned-proxy holdout scoring only; no MC1 outcome is read or joined",
        "proxy_root": str(proxy_root),
        "descriptor_roots": [str(root.resolve()) for root in args.descriptor_root],
        "holdout_plan": str(args.holdout_plan.resolve()),
        "heldout_trials": names,
        "selection_authority": "none; this is a frozen falsification prediction receipt",
    })
    _once(out / "correctness_report.json", {
        "all_scored_trials_are_predeclared_holdouts": True,
        "all_inputs_are_frozen_proxy_and_strict_oof_descriptor_receipts": True,
        "no_holdout_mc1_or_portfolio_output_was_read": True,
        "no_live_or_model_score_mutation": True,
    })
    print(out)


if __name__ == "__main__":
    main()
