#!/usr/bin/env python3
"""Fit a controlled meta ablation with train-frozen residual leaf composites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_global_residual_champion_enhancement import (
    DEFAULT_COMPACT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_JULY_SOURCE,
    DEFAULT_LEDGER,
    DEFAULT_REFERENCE_DIR,
    _daily_signed_autocorrelation,
    _fit_fixed_revision,
    _load_comparison_data,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    _reference_contract,
    metrics_by_scope,
)


COMPOSITES = [
    "residual_leaf_failure_probability_train_pct",
    "residual_leaf_time_risk_train_pct",
    "residual_leaf_feature_risk_train_pct",
    "residual_leaf_risk_composite_max",
    "residual_leaf_risk_composite_mean",
]


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    reference_features, params, reference_manifest = _reference_contract(
        args.reference_dir
    )
    data, data_manifest = _load_comparison_data(
        args.compact,
        args.ledger,
        args.july_source,
        args.feature_root,
        reference_features,
        data_start=pd.Timestamp(args.train_start, tz="UTC"),
        evaluation_end=pd.Timestamp(args.evaluation_end, tz="UTC"),
    )
    active_composites = (
        COMPOSITES[1:3] if str(args.composite_set) == "focused" else COMPOSITES
    )
    composite = pd.read_parquet(args.composites)
    composite["__ts__"] = pd.to_datetime(composite["__ts__"], utc=True)
    keys = ["__ts__", "side_name", "archetype_policy_key"]
    composite = composite[[*keys, *active_composites]].drop_duplicates(keys, keep="last")
    data = data.merge(composite, on=keys, how="inner", validate="many_to_one")
    fit_end = pd.Timestamp(args.fit_end, tz="UTC")
    eval_end = pd.Timestamp(args.evaluation_end, tz="UTC")
    train = data.loc[data["__ts__"].lt(fit_end)].reset_index(drop=True)
    evaluation = data.loc[
        data["__ts__"].ge(fit_end) & data["__ts__"].lt(eval_end)
    ].reset_index(drop=True)
    if len(train) < 5_000 or len(evaluation) < 1_000:
        raise ValueError(f"Insufficient ablation rows: train={len(train)}, eval={len(evaluation)}")
    baseline, _, baseline_manifest = _fit_fixed_revision(
        train,
        evaluation,
        reference_features,
        params,
        arm="leaf_matched_baseline",
        seed=int(args.seed),
    )
    arm_features = {
        "leaf_failure_probability": [COMPOSITES[0]],
        "leaf_time_risk": [COMPOSITES[1]],
        "leaf_feature_risk": [COMPOSITES[2]],
        "leaf_composite_max": [COMPOSITES[3]],
        "leaf_composite_mean": [COMPOSITES[4]],
        "leaf_time_plus_feature": [COMPOSITES[1], COMPOSITES[2]],
        "leaf_all_composites": COMPOSITES,
    }
    risk_weight_alpha: dict[str, float] = {}
    if str(args.composite_set) == "focused":
        alphas = [
            float(value)
            for value in str(args.risk_weight_alphas).split(",")
            if value.strip()
        ]
        arm_features = {}
        for alpha in alphas:
            suffix = str(alpha).replace(".", "p")
            arm = (
                "leaf_time_plus_feature"
                if alpha == 0.0
                else f"leaf_time_plus_feature_weight_{suffix}"
            )
            arm_features[arm] = active_composites
            risk_weight_alpha[arm] = alpha
    metrics_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    alternative_manifests: dict[str, object] = {}
    baseline_added = False
    for arm_index, (arm, added_features) in enumerate(arm_features.items()):
        alpha = float(risk_weight_alpha.get(arm, 0.0))
        sample_weight_multiplier = None
        if alpha > 0.0:
            risk = train[active_composites].max(axis=1).clip(0.0, 1.0)
            sample_weight_multiplier = (
                1.0 + alpha * np.square(risk.to_numpy(dtype=np.float32))
            ).astype(np.float32)
        alternative, _, alternative_manifest = _fit_fixed_revision(
            train,
            evaluation,
            [*reference_features, *added_features],
            params,
            arm=arm,
            seed=int(args.seed),
            sample_weight_multiplier=sample_weight_multiplier,
        )
        alternative["score_current_reference"] = baseline[
            "score_alternative"
        ].to_numpy(dtype=np.float32)
        alternative["hit_prob_current_reference"] = baseline[
            "hit_prob_alternative"
        ].to_numpy(dtype=np.float32)
        alternative["alternative_arm"] = arm
        metrics = metrics_by_scope(alternative, arm)
        if baseline_added:
            metrics = metrics.loc[~metrics["selector"].eq("current_reference")]
        else:
            baseline_added = True
        metrics_parts.append(metrics)
        alternative.to_parquet(
            args.output / f"oos_predictions__{arm}.parquet",
            index=False,
            compression="zstd",
        )
        alternative_manifests[arm] = alternative_manifest
        for selector in (("current_reference", arm) if arm_index == 0 else (arm,)):
            overall = metrics.loc[
                metrics["scope"].eq("overall")
                & metrics["fraction"].eq(0.10)
                & metrics["selector"].eq(selector)
            ].iloc[0]
            week = metrics.loc[
                metrics["scope"].eq("week")
                & metrics["fraction"].eq(0.10)
                & metrics["selector"].eq(selector)
            ]
            month = metrics.loc[
                metrics["scope"].eq("month")
                & metrics["fraction"].eq(0.10)
                & metrics["selector"].eq(selector)
            ]
            row = overall.to_dict()
            row["worst_week_ev"] = pd.to_numeric(
                week["mean_ev_after_1pct"], errors="coerce"
            ).min()
            row["worst_month_ev"] = pd.to_numeric(
                month["mean_ev_after_1pct"], errors="coerce"
            ).min()
            score_column = (
                "score_current_reference"
                if selector == "current_reference"
                else "score_alternative"
            )
            probability_column = (
                "hit_prob_current_reference"
                if selector == "current_reference"
                else "hit_prob_alternative"
            )
            row.update(
                _daily_signed_autocorrelation(
                    alternative, score_column, probability_column
                )
            )
            row["added_features"] = "|".join(
                [] if selector == "current_reference" else added_features
            )
            row["risk_weight_alpha"] = alpha if selector != "current_reference" else 0.0
            summary_rows.append(row)
    metrics = pd.concat(metrics_parts, ignore_index=True, copy=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    manifest = {
        "schema": "residual_leaf_meta_ablation_v1",
        "train_period": [args.train_start, args.fit_end],
        "evaluation_period": [args.fit_end, args.evaluation_end],
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(evaluation)),
        "composites": active_composites,
        "reference_manifest": reference_manifest,
        "data_manifest": data_manifest,
        "baseline_fit": baseline_manifest,
        "alternative_fits": alternative_manifests,
        "arm_features": arm_features,
        "comparison_contract": (
            "Baseline and alternative use identical rows, target, fixed production "
            "parameters, costs, and top-k selection. Only the declared train-frozen "
            "leaf composite columns differ by arm."
        ),
        "limitation": (
            "This first controlled ablation trains meta on January-March 2026 because "
            "those are the currently materialized leaf-OOF rows."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composites", type=Path, default=Path("data_perp/reports/residual_leaf_state_discovery_20260712_v4/oos_leaf_state_composites.parquet"))
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--july-source", type=Path, default=DEFAULT_JULY_SOURCE)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/residual_leaf_meta_ablation_20260712_v1"))
    parser.add_argument("--train-start", default="2026-01-01")
    parser.add_argument("--fit-end", default="2026-04-01")
    parser.add_argument("--evaluation-end", default="2026-07-11")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--composite-set", choices=("focused", "all"), default="all")
    parser.add_argument(
        "--risk-weight-alphas",
        default="0.0",
        help="Focused-mode high leaf-risk training multipliers to compare.",
    )
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": "complete", "output": str(args.output), "train_rows": manifest["train_rows"], "evaluation_rows": manifest["evaluation_rows"]}, indent=2))


if __name__ == "__main__":
    main()
