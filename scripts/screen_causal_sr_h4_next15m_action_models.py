#!/usr/bin/env python3
"""Strict-prior model screen for sparse next-15m exit action values.

This is deliberately a *screen*, not a policy promotion runner.  It compares
small causal regressors on already-frozen exact temporary-action labels and
reports only strict-prior monthly OOF label metrics.  A model must subsequently
pass the expensive exact constrained-portfolio replay and a frozen 2026 check.
No live/exchange module is imported or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from scripts import run_causal_sr_h4_actuator_counterfactual_ablation as base
    from scripts import run_causal_sr_h4_next15m_actuator_ablation as next15
except ModuleNotFoundError:
    import run_causal_sr_h4_actuator_counterfactual_ablation as base
    import run_causal_sr_h4_next15m_actuator_ablation as next15


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


CONFIGS = (
    {"name": "l2_d3_l7_c05_l40", "objective": "regression_l2", "depth": 3, "leaves": 7, "child_fraction": .05, "reg_lambda": 40., "positive_weight": 1.},
    {"name": "l1_d2_l4_c10_l80", "objective": "regression_l1", "depth": 2, "leaves": 4, "child_fraction": .10, "reg_lambda": 80., "positive_weight": 1.},
    {"name": "l1_d3_l7_c05_l80", "objective": "regression_l1", "depth": 3, "leaves": 7, "child_fraction": .05, "reg_lambda": 80., "positive_weight": 1.},
    {"name": "huber_d3_l7_c05_l80", "objective": "huber", "depth": 3, "leaves": 7, "child_fraction": .05, "reg_lambda": 80., "positive_weight": 1.},
    {"name": "l2_d2_l4_c15_l80", "objective": "regression_l2", "depth": 2, "leaves": 4, "child_fraction": .15, "reg_lambda": 80., "positive_weight": 1.},
    {"name": "l2_d4_l15_c05_l80", "objective": "regression_l2", "depth": 4, "leaves": 15, "child_fraction": .05, "reg_lambda": 80., "positive_weight": 1.},
    {"name": "l2_d3_l7_c05_l80_pw4", "objective": "regression_l2", "depth": 3, "leaves": 7, "child_fraction": .05, "reg_lambda": 80., "positive_weight": 4.},
    {"name": "l1_d3_l7_c05_l80_pw4", "objective": "regression_l1", "depth": 3, "leaves": 7, "child_fraction": .05, "reg_lambda": 80., "positive_weight": 4.},
)


def _fit(train: pd.DataFrame, fields: tuple[str, ...], target_column: str, config: dict[str, object]) -> lgb.LGBMRegressor:
    child = max(64, int(np.ceil(len(train) * float(config["child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective=str(config["objective"]), n_estimators=360, learning_rate=.03,
        max_depth=int(config["depth"]), num_leaves=int(config["leaves"]), min_child_samples=child,
        subsample=.8, colsample_bytree=.8, reg_lambda=float(config["reg_lambda"]),
        random_state=1729, n_jobs=2, verbosity=-1,
    )
    target = train[target_column].to_numpy(float)
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    weights *= np.where(target > 0.0, float(config["positive_weight"]), 1.0)
    model.fit(train.loc[:, fields], target, sample_weight=weights)
    return model


def _metrics(values: pd.DataFrame, target_column: str) -> dict[str, float]:
    result: dict[str, float] = {
        "oof_rows": float(len(values)),
        "oof_states": float(values[["candidate_id", "state_decision_ts"]].drop_duplicates().shape[0]),
        "spearman": float(values["score"].corr(values[target_column], method="spearman")),
    }
    for pct in (1, 2, 5, 10):
        count = max(1, int(np.ceil(len(values) * pct / 100.0)))
        top = values.nlargest(count, "score")[target_column]
        result[f"top{pct}_mean_advantage_bps"] = float(top.mean())
        result[f"top{pct}_positive_share"] = float((top > 0.0).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, default=base.DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATES)
    parser.add_argument("--multiplier", type=float, default=.65, help="Temporary multiplier label to screen; must exist in the immutable label grid.")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    label_root = args.labels_root.resolve()
    labels = pd.read_parquet(label_root / "next15_counterfactual_labels.parquet")
    if set(labels["actuator"].astype(str)).__len__() != 1:
        raise AssertionError("labels must contain one actuator")
    next15.MULTIPLIERS = tuple(sorted(pd.to_numeric(labels["multiplier"], errors="raise").astype(float).unique()))
    if not any(np.isclose(value, 1.0) for value in next15.MULTIPLIERS):
        raise AssertionError("label grid has no neutral parent multiplier")
    if not any(np.isclose(value, float(args.multiplier)) for value in next15.MULTIPLIERS):
        raise AssertionError("requested multiplier is absent from the immutable label grid")
    target_column = next15._label_column(float(args.multiplier))
    _, _, _, _, states = base._load_parent(args.parent_root.resolve(), args.state_root.resolve())
    # Keep the original target-free state contract.  The joined panel contains
    # temporary action labels and must never define learner features.
    fields = base._state_fields(states)
    panel = next15._label_states(states, labels)
    start, end = _month("2025-06"), _month("2026-01")
    panel = panel.loc[panel["entry_decision_ts"].ge(start) & panel["entry_decision_ts"].lt(end)].copy()
    records: list[pd.DataFrame] = []
    for config in CONFIGS:
        pieces: list[pd.DataFrame] = []
        for held in pd.period_range(start, end - pd.offsets.MonthBegin(1), freq="M"):
            held_start = pd.Timestamp(held.start_time, tz="UTC")
            held_end = held_start + pd.offsets.MonthBegin(1)
            train = panel.loc[
                panel["entry_decision_ts"].ge(start) & panel["entry_decision_ts"].lt(held_start)
                & panel["policy_label_available_ts"].lt(held_start)
            ].copy()
            test = panel.loc[panel["entry_decision_ts"].ge(held_start) & panel["entry_decision_ts"].lt(held_end)].copy()
            if train["candidate_id"].nunique() < 250 or test.empty:
                continue
            score = _fit(train, fields, target_column, config).predict(test.loc[:, fields])
            pieces.append(test.loc[:, ["candidate_id", "state_decision_ts", "entry_decision_ts", target_column]].assign(model=config["name"], held_month=held_start.strftime("%Y-%m"), score=score))
        if not pieces:
            raise RuntimeError(f"{config['name']} produced no strict-prior OOF predictions")
        records.append(pd.concat(pieces, ignore_index=True))
    predictions = pd.concat(records, ignore_index=True)
    summary = pd.DataFrame([
        {"model": model, **_metrics(group, target_column)} for model, group in predictions.groupby("model", sort=True)
    ]).sort_values(["top2_mean_advantage_bps", "top5_mean_advantage_bps", "spearman"], ascending=False, kind="stable")
    monthly = pd.DataFrame([
        {"model": model, "held_month": month, **_metrics(group, target_column)}
        for (model, month), group in predictions.groupby(["model", "held_month"], sort=True)
    ])
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(out / "2025_strict_prior_action_value_predictions.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "2025_strict_prior_action_model_summary.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "2025_strict_prior_action_model_monthly_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(CONFIGS).to_parquet(out / "model_configurations.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(fields)), "feature": fields}).to_parquet(out / "target_free_feature_contract.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-h4-next15m-action-model-screen-v1",
        "scope": "offline label-only model screen; no live, exchange, policy, admission, portfolio, geometry, C1 S/R, or MC1 mutation",
        "target": f"temporary {args.multiplier:g}x next-15m action advantage in exact H12 net bps",
        "selection": "strict-prior monthly 2025-06..2025-12 OOF; no 2026 target accessed",
        "models": CONFIGS,
        "labels_root": str(label_root),
        "labels_manifest_sha256": _sha(label_root / "run_manifest.json"),
        "parent_root": str(args.parent_root.resolve()),
        "parent_manifest_sha256": _sha(args.parent_root.resolve() / "run_manifest.json"),
        "state_root": str(args.state_root.resolve()),
        "state_manifest_sha256": _sha(args.state_root.resolve() / "run_manifest.json"),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
