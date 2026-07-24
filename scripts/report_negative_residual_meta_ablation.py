#!/usr/bin/env python3
"""Report production-meta versus hardened negative-residual feature ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
)


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _load(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    keep = KEYS + [
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "score_alternative",
        "policy_selected_alternative",
    ]
    return frame[keep].rename(
        columns={name: f"{name}_{suffix}" for name in keep if name not in KEYS}
    )


def _summary(frame: pd.DataFrame, selector: str, scope: str, value: str) -> dict[str, object]:
    selected = frame[f"policy_selected_alternative_{selector}"].fillna(False).astype(bool)
    rows = frame.loc[selected]
    score = pd.to_numeric(frame[f"score_alternative_{selector}"], errors="coerce").clip(1e-6, 1 - 1e-6)
    target = pd.to_numeric(frame[f"clean_exec_{selector}"], errors="coerce").fillna(0).clip(0, 1)
    valid = score.notna() & target.notna()
    days = max(frame["__ts__"].dt.floor("D").nunique(), 1)
    return {
        "selector": selector,
        "scope": scope,
        "scope_value": value,
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(rows)),
        "trades_per_day": float(len(rows) / days),
        "mean_ev_after_1pct": float(pd.to_numeric(rows[f"ev_after_1pct_{selector}"], errors="coerce").mean()),
        "sum_ev_after_1pct": float(pd.to_numeric(rows[f"ev_after_1pct_{selector}"], errors="coerce").sum()),
        "positive_ev_rate": float(pd.to_numeric(rows[f"ev_after_1pct_{selector}"], errors="coerce").gt(0).mean()),
        "clean_exec_precision": float(pd.to_numeric(rows[f"clean_exec_{selector}"], errors="coerce").mean()),
        "dirty_positive_rate": float(pd.to_numeric(rows[f"dirty_positive_{selector}"], errors="coerce").mean()),
        "first_touch_bad_mae_rate": float(pd.to_numeric(rows[f"first_touch_bad_mae_1r_{selector}"], errors="coerce").mean()),
        "full_path_bad_mae_rate": float(pd.to_numeric(rows[f"full_path_bad_mae_1r_{selector}"], errors="coerce").mean()),
        "timeout_rate": float(pd.to_numeric(rows[f"timeout_{selector}"], errors="coerce").mean()),
        "brier_clean_exec": float(np.mean((score.loc[valid] - target.loc[valid]) ** 2)),
        "log_loss_clean_exec": float(log_loss(target.loc[valid], score.loc[valid], labels=[0, 1])),
        "pr_auc_clean_exec": float(average_precision_score(target.loc[valid], score.loc[valid])),
    }


def _scoped_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "all", frame)]
    work = frame.assign(calendar_month=frame["__ts__"].dt.to_period("M").astype(str))
    for month, local in work.groupby("calendar_month", observed=True):
        scopes.append(("month", str(month), local))
    for side, local in work.groupby("side_name", observed=True):
        scopes.append(("side", str(side), local))
    for (side, archetype), local in work.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        scopes.append(("side_archetype", f"{side}__{archetype}", local))
    for scope, value, local in scopes:
        for selector in ("baseline", "augmented"):
            rows.append(_summary(local, selector, scope, value))
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics.selector.eq("baseline")].set_index(["scope", "scope_value"])
    augmented = metrics.loc[metrics.selector.eq("augmented")].set_index(["scope", "scope_value"])
    numeric = [name for name in metrics.columns if name not in {"selector", "scope", "scope_value"}]
    delta = augmented[numeric].subtract(baseline[numeric]).reset_index()
    delta.insert(0, "selector", "augmented_minus_baseline")
    return pd.concat([metrics, delta], ignore_index=True, sort=False)


def _episode_metrics(frame: pd.DataFrame, calendar_path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(calendar_path)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    work = frame.assign(day=frame["__ts__"].dt.floor("D"))
    rows = []
    for _, cell in calendar.iterrows():
        local = work.loc[
            work["day"].eq(cell["day"])
            & work["side_name"].eq(cell["side_name"])
            & work["archetype_policy_key"].eq(cell["archetype_policy_key"])
        ]
        if local.empty:
            continue
        record = {
            "day": cell["day"],
            "side_name": cell["side_name"],
            "archetype_policy_key": cell["archetype_policy_key"],
            "prior_status": cell.get("status", "unknown"),
            "candidate_rows": int(len(local)),
        }
        for selector in ("baseline", "augmented"):
            selected = local[f"policy_selected_alternative_{selector}"].fillna(False).astype(bool)
            chosen = local.loc[selected]
            record[f"selected_rows_{selector}"] = int(len(chosen))
            record[f"mean_ev_{selector}"] = float(chosen[f"ev_after_1pct_{selector}"].mean())
            record[f"clean_precision_{selector}"] = float(chosen[f"clean_exec_{selector}"].mean())
            record[f"bad_mae_{selector}"] = float(chosen[f"first_touch_bad_mae_1r_{selector}"].mean())
        record["delta_mean_ev"] = record["mean_ev_augmented"] - record["mean_ev_baseline"]
        record["delta_clean_precision"] = record["clean_precision_augmented"] - record["clean_precision_baseline"]
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["day", "side_name", "archetype_policy_key"])


def _tree_usage(model_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    booster = model.booster_
    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total_gain = max(float(gain.sum()), 1e-12)
    rows = pd.DataFrame(
        {
            "feature": names,
            "gain": gain,
            "gain_share": gain / total_gain,
            "split_count": split,
        }
    )
    new_keys = set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    rows["feature_group"] = np.where(
        rows.feature.isin(new_keys),
        "new_market_context",
        np.where(rows.feature.str.startswith("residual_state_family_"), "new_family_context", "production_reference"),
    )
    rows["used"] = rows["split_count"].gt(0)
    return rows.sort_values(["gain", "split_count"], ascending=False, kind="stable")


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = _load(args.baseline, "baseline")
    augmented = _load(args.augmented, "augmented")
    aligned = baseline.merge(augmented, on=KEYS, how="inner", validate="one_to_one")
    aligned.to_parquet(args.output / "aligned_oos_rows.parquet", index=False, compression="zstd")
    metrics = _scoped_metrics(aligned)
    metrics.to_csv(args.output / "metrics_global_month_side_archetype.csv", index=False)
    episodes = _episode_metrics(aligned, args.calendar)
    episodes.to_csv(args.output / "episode_metrics.csv", index=False)
    usage = _tree_usage(args.model)
    usage.to_csv(args.output / "new_feature_tree_usage.csv", index=False)
    fold_usage_parts = []
    fold_model_dir = args.model.parent / "fold_models"
    for fold_model in sorted(fold_model_dir.glob("score_*.joblib")):
        local = _tree_usage(fold_model)
        local.insert(0, "fold", fold_model.stem.removeprefix("score_"))
        fold_usage_parts.append(local)
    fold_usage = (
        pd.concat(fold_usage_parts, ignore_index=True)
        if fold_usage_parts
        else pd.DataFrame()
    )
    if not fold_usage.empty:
        fold_usage.to_csv(args.output / "feature_tree_usage_by_fold.csv", index=False)
        stability = (
            fold_usage.groupby(["feature", "feature_group"], observed=True)
            .agg(
                folds=("fold", "nunique"),
                folds_used=("used", "sum"),
                mean_gain_share=("gain_share", "mean"),
                max_gain_share=("gain_share", "max"),
                total_splits=("split_count", "sum"),
            )
            .reset_index()
            .sort_values(["folds_used", "mean_gain_share"], ascending=False)
        )
        stability.to_csv(args.output / "feature_usage_stability.csv", index=False)
    grouped_usage = usage.groupby("feature_group", observed=True).agg(
        features=("feature", "size"),
        used_features=("used", "sum"),
        gain_share=("gain_share", "sum"),
        split_count=("split_count", "sum"),
    ).reset_index()
    grouped_usage.to_csv(args.output / "feature_group_usage.csv", index=False)
    overall = metrics.loc[(metrics.scope == "overall") & (metrics.scope_value == "all")]
    manifest = {
        "schema": "negative_residual_meta_ablation_report_v1",
        "aligned_oos_rows": int(len(aligned)),
        "episode_cells_in_oos_scope": int(len(episodes)),
        "tree_model": str(args.model),
        "fold_models_audited": int(len(fold_usage_parts)),
        "feature_group_usage": grouped_usage.to_dict("records"),
        "overall": overall.to_dict("records"),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
