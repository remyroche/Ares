#!/usr/bin/env python3
"""Select the GateProxy from pre-holdout grouped-validation evidence only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


SCHEMA = "strict_r3_p8u_meta_gateproxy_preholdout_selection_v1"
REQUIRED_VALIDATIONS = ("target_family", "loss", "era")


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.fit_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    correctness = json.loads((root / "correctness_report.json").read_text())
    required_receipt = {
        "descriptors_are_strict_oof_and_target_free_before_outcome_metrics",
        "downstream_labels_are_matched_six_month_mc1",
        "grouped_validation_includes_supported_target_loss_and_era_holdouts",
        "leave_feature_contract_out_is_explicitly_reported",
        "portfolio_is_not_the_sole_proxy_training_target",
        "priority_and_gate_proxies_are_separate",
        "proxy_has_no_direct_live_or_model_score_authority",
    }
    if not all(correctness.get(key) is True for key in required_receipt):
        raise AssertionError("fit correctness receipt is incomplete")
    metric = pd.read_parquet(root / "proxy_grouped_cv_metrics.parquet")
    gate = metric.loc[(metric.target.eq("dgate_shrunk")) & metric.validation.isin(REQUIRED_VALIDATIONS)].copy()
    rows: list[dict[str, object]] = []
    for model, group in gate.groupby("model", sort=True):
        means = group.groupby("validation", sort=True).spearman.mean()
        if set(means.index) != set(REQUIRED_VALIDATIONS) or not means.notna().all():
            raise AssertionError(f"{model}: incomplete supported GateProxy validation")
        rows.append({
            "model": model,
            **{f"spearman_{key}": float(means[key]) for key in REQUIRED_VALIDATIONS},
            "min_supported_spearman": float(means.min()),
            "mean_supported_spearman": float(means.mean()),
        })
    result = pd.DataFrame(rows).sort_values(
        ["min_supported_spearman", "mean_supported_spearman", "model"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    selected = result.iloc[0].to_dict()
    out.mkdir(parents=True)
    result.to_parquet(out / "preholdout_gateproxy_model_comparison.parquet", index=False, compression="zstd")
    _once(out / "preholdout_gateproxy_choice.json", {
        "schema": SCHEMA,
        "scope": "offline selection of a learned GateProxy surrogate only; no HPO trial, MC1, portfolio, or live mutation",
        "fit_root": str(root),
        "selection": "maximize minimum supported grouped-holdout Spearman, then mean supported Spearman",
        "supported_holdouts": list(REQUIRED_VALIDATIONS),
        "selected": selected,
        "selection_authority": "shortlist ranking only; full MC1 remains the sole Meta-trial promotion authority",
    })
    _once(out / "correctness_report.json", {
        "selection_uses_only_preholdout_grouped_validation_metrics": True,
        "priority_proxy_is_not_considered": True,
        "all_supported_gate_holdouts_are_used": True,
    })
    print(out)


if __name__ == "__main__":
    main()
