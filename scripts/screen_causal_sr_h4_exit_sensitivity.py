#!/usr/bin/env python3
"""Strict-prior screen for temporary-action exit sensitivity (offline only).

The direct target is the exact net-bps advantage from a temporary next-15m
action.  The decomposed control instead estimates:

    P(the action changes the realised exit path | causal state)
    × E(exact action advantage | exit path changed, causal state)

All labels become eligible only after the complete H12 policy path resolves.
This is a label-level information test: it never changes exit policy, model
admission, portfolio construction, Geometry/K9, or any live component.
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
except ModuleNotFoundError:
    import run_causal_sr_h4_actuator_counterfactual_ablation as base


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _weight(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 / frame.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)


def _direct_model(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=360, learning_rate=.03,
        max_depth=3, num_leaves=7, min_child_samples=max(64, int(np.ceil(len(train) * .05)),),
        subsample=.8, colsample_bytree=.8, reg_lambda=80., random_state=1729, n_jobs=2, verbosity=-1,
    )
    target = train["advantage_bps"].to_numpy(float)
    weights = _weight(train) * np.where(target > 0.0, 4.0, 1.0)
    model.fit(train.loc[:, fields], target, sample_weight=weights)
    return model


def _change_model(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=.03,
        max_depth=3, num_leaves=7, min_child_samples=max(64, int(np.ceil(len(train) * .05))),
        subsample=.8, colsample_bytree=.8, reg_lambda=80., random_state=1729, n_jobs=2, verbosity=-1,
    )
    target = train["exit_changed"].astype(int).to_numpy()
    weights = _weight(train)
    # Balance only within the causal training fold; this is not an outcome
    # feature and makes sparse-event probabilities learnable.
    positive = max(int(target.sum()), 1)
    negative = max(int((1 - target).sum()), 1)
    weights *= np.where(target == 1, negative / positive, 1.0)
    model.fit(train.loc[:, fields], target, sample_weight=weights)
    return model


def _conditional_model(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMRegressor | None:
    changed = train.loc[train["exit_changed"].astype(bool)].copy()
    if changed["candidate_id"].nunique() < 100:
        return None
    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=280, learning_rate=.03,
        max_depth=3, num_leaves=7, min_child_samples=max(32, int(np.ceil(len(changed) * .05))),
        subsample=.8, colsample_bytree=.8, reg_lambda=80., random_state=1729, n_jobs=2, verbosity=-1,
    )
    target = changed["advantage_bps"].to_numpy(float)
    weights = _weight(changed) * np.where(target > 0.0, 2.0, 1.0)
    model.fit(changed.loc[:, fields], target, sample_weight=weights)
    return model


def _tail_metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    out: dict[str, float] = {
        "rows": float(len(frame)),
        "states": float(frame[["candidate_id", "state_decision_ts"]].drop_duplicates().shape[0]),
        "advantage_spearman": float(frame[score].corr(frame["advantage_bps"], method="spearman")),
        "exit_changed_spearman": float(frame[score].corr(frame["exit_changed"].astype(float), method="spearman")),
    }
    for pct in (1, 2, 5, 10):
        count = max(1, int(np.ceil(len(frame) * pct / 100.0)))
        tail = frame.nlargest(count, score)
        changed = tail.loc[tail["exit_changed"].astype(bool), "advantage_bps"]
        out[f"top{pct}_mean_advantage_bps"] = float(tail["advantage_bps"].mean())
        out[f"top{pct}_positive_share"] = float((tail["advantage_bps"] > 0.0).mean())
        out[f"top{pct}_changed_share"] = float(tail["exit_changed"].mean())
        out[f"top{pct}_mean_advantage_if_changed_bps"] = float(changed.mean()) if not changed.empty else np.nan
    return out


def _calibration(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    source = frame.loc[:, ["candidate_id", "state_decision_ts", score, "advantage_bps", "exit_changed"]].copy()
    source["bin"] = pd.qcut(source[score].rank(method="first"), q=10, labels=False, duplicates="drop")
    return source.groupby("bin", as_index=False).agg(
        rows=("candidate_id", "size"),
        mean_predicted_score=(score, "mean"),
        mean_realised_advantage_bps=("advantage_bps", "mean"),
        exit_changed_share=("exit_changed", "mean"),
    ).assign(score=score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--action-multiplier", type=float, required=True)
    parser.add_argument("--parent-root", type=Path, default=base.DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATES)
    parser.add_argument("--extra-feature-panel", type=Path, default=None,
                        help="Optional target-free state panel keyed by candidate_id/state_decision_ts. "
                             "The run emits both the unchanged 91-field control and control-plus-extra.")
    parser.add_argument("--extra-feature-columns", type=str, default=None,
                        help="Optional comma-separated subset of numeric columns from --extra-feature-panel. "
                             "Useful for predeclared within-block ablations; never selects using labels.")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    labels_root = args.labels_root.resolve()
    labels = pd.read_parquet(labels_root / "next15_counterfactual_labels.parquet")
    required = {
        "candidate_id", "state_decision_ts", "multiplier", "advantage_bps", "exit_changed",
        "policy_label_available_ts", "actuator",
    }
    if missing := required.difference(labels.columns):
        raise AssertionError(f"exit-sensitivity label receipt lacks {sorted(missing)}")
    labels["state_decision_ts"] = pd.to_datetime(labels["state_decision_ts"], utc=True, errors="raise")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="raise")
    labels = labels.loc[np.isclose(pd.to_numeric(labels["multiplier"], errors="raise"), float(args.action_multiplier))].copy()
    if labels.empty or not labels["exit_changed"].astype(bool).any():
        raise RuntimeError("requested action has no changed-exit labels")
    if labels.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("action label duplicates a state")
    _, _, _, _, states = base._load_parent(args.parent_root.resolve(), args.state_root.resolve())
    control_fields = base._state_fields(states)
    feature_sets: dict[str, tuple[str, ...]] = {"control_91": control_fields}
    extra_columns: tuple[str, ...] = ()
    if args.extra_feature_panel is not None:
        extra_path = args.extra_feature_panel.resolve()
        extra = pd.read_parquet(extra_path).copy()
        key = ["candidate_id", "state_decision_ts"]
        if set(key).difference(extra.columns):
            raise AssertionError("extra feature panel lacks state identity")
        extra["candidate_id"] = extra["candidate_id"].astype(str)
        extra["state_decision_ts"] = pd.to_datetime(extra["state_decision_ts"], utc=True, errors="raise")
        if extra.duplicated(key).any():
            raise AssertionError("extra feature panel duplicates state identity")
        expected = states.loc[:, key].copy()
        if len(extra) != len(expected) or not extra.loc[:, key].sort_values(key).reset_index(drop=True).equals(
            expected.sort_values(key).reset_index(drop=True)
        ):
            raise AssertionError("extra feature panel is not exactly aligned to the target-free H4 state identity")
        extra_columns = tuple(
            str(field) for field in extra.columns
            if field not in key and pd.api.types.is_numeric_dtype(extra[field])
        )
        if args.extra_feature_columns is not None:
            requested = tuple(field.strip() for field in args.extra_feature_columns.split(",") if field.strip())
            unavailable = set(requested).difference(extra_columns)
            if unavailable:
                raise AssertionError(f"requested extra feature columns are not numeric panel fields: {sorted(unavailable)}")
            extra_columns = requested
        if not extra_columns:
            raise AssertionError("extra feature panel has no numeric target-free fields")
        conflict = set(extra_columns).intersection(control_fields)
        if conflict:
            raise AssertionError(f"extra feature fields overlap the 91-field control: {sorted(conflict)}")
        states = states.merge(extra.loc[:, [*key, *extra_columns]], on=key, how="left", validate="one_to_one")
        feature_sets["control_91_plus_extra"] = (*control_fields, *extra_columns)
    panel = states.merge(
        labels.loc[:, ["candidate_id", "state_decision_ts", "advantage_bps", "exit_changed", "policy_label_available_ts"]],
        on=["candidate_id", "state_decision_ts"], how="inner", validate="one_to_one",
    )
    start, end = _month("2025-06"), _month("2026-01")
    panel = panel.loc[panel["entry_decision_ts"].ge(start) & panel["entry_decision_ts"].lt(end)].copy()
    pieces: list[pd.DataFrame] = []
    for model_variant, fields in feature_sets.items():
        for month in pd.period_range(start, end - pd.offsets.MonthBegin(1), freq="M"):
            held_start = pd.Timestamp(month.start_time, tz="UTC")
            held_end = held_start + pd.offsets.MonthBegin(1)
            train = panel.loc[
                panel["entry_decision_ts"].ge(start) & panel["entry_decision_ts"].lt(held_start)
                & panel["policy_label_available_ts"].lt(held_start)
            ].copy()
            test = panel.loc[panel["entry_decision_ts"].ge(held_start) & panel["entry_decision_ts"].lt(held_end)].copy()
            if test.empty or train["candidate_id"].nunique() < 250 or train["exit_changed"].astype(bool).sum() < 100:
                continue
            direct = _direct_model(train, fields)
            change = _change_model(train, fields)
            conditional = _conditional_model(train, fields)
            block = test.loc[:, ["candidate_id", "state_decision_ts", "entry_decision_ts", "advantage_bps", "exit_changed"]].copy()
            block["direct_advantage_score"] = direct.predict(test.loc[:, fields])
            block["pred_exit_changed_probability"] = change.predict_proba(test.loc[:, fields])[:, 1]
            if conditional is None:
                block["conditional_advantage_if_changed_score"] = 0.0
            else:
                block["conditional_advantage_if_changed_score"] = conditional.predict(test.loc[:, fields])
            block["decomposed_expected_advantage_score"] = block["pred_exit_changed_probability"] * block["conditional_advantage_if_changed_score"]
            block["held_month"] = held_start.strftime("%Y-%m")
            block["model_variant"] = model_variant
            pieces.append(block)
    if not pieces:
        raise RuntimeError("no strict-prior exit-sensitivity predictions")
    predictions = pd.concat(pieces, ignore_index=True)
    metrics = []
    for model_variant, variant in predictions.groupby("model_variant", sort=True):
        for score in ("direct_advantage_score", "decomposed_expected_advantage_score", "pred_exit_changed_probability"):
            metrics.append({"model_variant": model_variant, "score": score, **_tail_metrics(variant, score)})
    monthly_records: list[dict[str, object]] = []
    for (model_variant, month), group in predictions.groupby(["model_variant", "held_month"], sort=True):
        for score in ("direct_advantage_score", "decomposed_expected_advantage_score", "pred_exit_changed_probability"):
            monthly_records.append({"model_variant": model_variant, "score": score, "held_month": month, **_tail_metrics(group, score)})
    monthly = pd.DataFrame(monthly_records)
    calibration = pd.concat([
        _calibration(variant, score).assign(model_variant=model_variant)
        for model_variant, variant in predictions.groupby("model_variant", sort=True)
        for score in ("direct_advantage_score", "decomposed_expected_advantage_score", "pred_exit_changed_probability")
    ], ignore_index=True)
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(out / "2025_strict_prior_exit_sensitivity_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(out / "2025_strict_prior_exit_sensitivity_summary.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "2025_strict_prior_exit_sensitivity_monthly_metrics.parquet", index=False, compression="zstd")
    calibration.to_parquet(out / "2025_strict_prior_exit_sensitivity_calibration.parquet", index=False, compression="zstd")
    pd.DataFrame([
        {"model_variant": name, "position": position, "feature": field}
        for name, fields in feature_sets.items() for position, field in enumerate(fields)
    ]).to_parquet(out / "target_free_h4_feature_contract.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-h4-exit-sensitivity-screen-v1",
        "scope": "offline causal label screen only; no exchange, live policy, admission, portfolio, MC1, C1 S/R, or Geometry/K9 mutation",
        "action_multiplier": float(args.action_multiplier),
        "targets": {
            "exit_changed": "counterfactual temporary action changes exact exit minute, reason, or exit price against the parent",
            "conditional_advantage": "exact temporary-action net-bps advantage conditional on exit_changed",
            "decomposed_expected_advantage": "P(exit_changed) × E(advantage | exit_changed)",
        },
        "selection": "strict-prior monthly OOF on 2025-06..2025-12; labels must resolve before each held month",
        "features": {"control_91": list(control_fields), "extra_target_free_fields": list(extra_columns)},
        "extra_feature_panel": str(args.extra_feature_panel.resolve()) if args.extra_feature_panel is not None else None,
        "labels_root": str(labels_root),
        "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "parent_root": str(args.parent_root.resolve()),
        "parent_manifest_sha256": _sha256(args.parent_root.resolve() / "run_manifest.json"),
        "state_root": str(args.state_root.resolve()),
        "state_manifest_sha256": _sha256(args.state_root.resolve() / "run_manifest.json"),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
