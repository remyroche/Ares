#!/usr/bin/env python3
"""Select a GateProxy only from support-aware grouped portability evidence.

This v2 selector is intentionally separate from the historic parent-54
pre-holdout selector.  It makes the newly available feature-contract holdout
first-class, but does not treat a three-trial contract as reliable evidence.
The selected model has *shortlist* authority only: every suggested Meta trial
still needs a fresh strict six-month MC1 confirmation before promotion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


SCHEMA = "strict_r3_p8u_meta_gateproxy_grouped_portability_selection_v2"
REQUIRED_VALIDATIONS = ("target_family", "loss", "feature_contract", "era")


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--minimum-held-trials", type=int, default=6,
        help="minimum observations in an individual holdout group to count as portability evidence",
    )
    args = parser.parse_args()
    if args.minimum_held_trials < 3:
        raise ValueError("--minimum-held-trials must be at least 3")
    root, out = args.fit_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    correctness = json.loads((root / "correctness_report.json").read_text())
    required_receipt = {
        "descriptors_are_strict_oof_and_target_free_before_outcome_metrics",
        "downstream_labels_are_matched_six_month_mc1",
        "grouped_validation_includes_supported_target_loss_and_era_holdouts",
        "leave_feature_contract_out_is_explicitly_reported",
        "leave_feature_contract_out_supported",
        "portfolio_is_not_the_sole_proxy_training_target",
        "priority_and_gate_proxies_are_separate",
        "proxy_has_no_direct_live_or_model_score_authority",
    }
    if not all(correctness.get(key) is True for key in required_receipt):
        raise AssertionError("fit correctness receipt is incomplete")
    metrics = pd.read_parquet(root / "proxy_grouped_cv_metrics.parquet")
    gate = metrics.loc[
        metrics.target.eq("dgate_shrunk") & metrics.validation.isin(REQUIRED_VALIDATIONS)
    ].copy()
    gate["eligible"] = gate.rows.ge(args.minimum_held_trials) & gate.spearman.notna()
    summaries: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for model, group in gate.groupby("model", sort=True):
        values: dict[str, float | int | str] = {"model": str(model)}
        for validation in REQUIRED_VALIDATIONS:
            part = group.loc[group.validation.eq(validation)].copy()
            usable = part.loc[part.eligible].copy()
            if usable.empty:
                raise AssertionError(f"{model}: no support-qualified {validation} holdout")
            values[f"groups_{validation}"] = int(len(part))
            values[f"eligible_groups_{validation}"] = int(len(usable))
            values[f"spearman_{validation}"] = float(usable.spearman.mean())
            values[f"min_group_spearman_{validation}"] = float(usable.spearman.min())
            values[f"top3_precision_{validation}"] = float(usable.top3_precision.mean())
            values[f"regret_at3_{validation}"] = float(usable.regret_at3.mean())
            for row in part.loc[~part.eligible].itertuples(index=False):
                excluded.append({
                    "model": str(model), "validation": str(validation),
                    "held_group": str(row.held_group), "rows": int(row.rows),
                    "reason": f"below minimum held trials ({args.minimum_held_trials}) or undefined Spearman",
                })
        support_means = [float(values[f"spearman_{validation}"]) for validation in REQUIRED_VALIDATIONS]
        values["min_supported_spearman"] = float(min(support_means))
        values["mean_supported_spearman"] = float(sum(support_means) / len(support_means))
        summaries.append(values)
    comparison = pd.DataFrame(summaries).sort_values(
        ["min_supported_spearman", "mean_supported_spearman", "model"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    selected = comparison.iloc[0].to_dict()
    out.mkdir(parents=True)
    comparison.to_parquet(out / "gateproxy_grouped_portability_comparison.parquet", index=False, compression="zstd")
    pd.DataFrame(excluded).to_parquet(out / "gateproxy_excluded_small_holdouts.parquet", index=False, compression="zstd")
    _once(out / "gateproxy_grouped_portability_choice.json", {
        "schema": SCHEMA,
        "scope": "offline selection of a learned GateProxy surrogate only; no HPO trial, MC1, portfolio, or live mutation",
        "fit_root": str(root),
        "selection": "maximize minimum support-qualified grouped-holdout Spearman, then mean across target/loss/feature-contract/era",
        "required_holdouts": list(REQUIRED_VALIDATIONS),
        "minimum_held_trials_per_group": int(args.minimum_held_trials),
        "small_holdout_policy": "record but do not use groups below minimum support to select the surrogate",
        "selected": selected,
        "selection_authority": "shortlist ranking only; full MC1 remains the sole Meta-trial promotion authority",
    })
    _once(out / "correctness_report.json", {
        "selection_uses_only_grouped_strict_oof_validation_metrics": True,
        "priority_proxy_is_not_considered_for_hpo_authority": True,
        "target_loss_feature_contract_and_era_holdouts_are_all_used": True,
        "small_feature_contract_holdouts_are_explicitly_excluded_from_selection": bool(excluded),
        "no_trial_or_live_model_promotion_or_mutation": True,
    })
    print(out)


if __name__ == "__main__":
    main()
