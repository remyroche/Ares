#!/usr/bin/env python3
"""Report fixed-contract breakout path context meta ablations.

The report evaluates the same global timestamp top-10% budget for every arm,
then isolates the short-breakout rows affected by the new fields.  It never
uses path outcomes as model inputs; rapid-reversal and retention incidence are
joined only after OOS scores have been emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data_perp/reports/breakout_path_quality_meta_ablation_20260713_v1")
TARGET_ARCHETYPE = "short_breakout_precision"
KEYS = ("__ts__", "__symbol__", "side_name")
PATH_LABEL_TARGETS = ("rapid_reversal", "severe_retention")


def _read_arm(path: Path) -> pd.DataFrame:
    shards = sorted((path / "prediction_shards").glob("*.parquet"))
    if not shards:
        combined = path / "s52_train_meta_regime_handoff_smoke_predictions.parquet"
        if combined.exists():
            shards = [combined]
    if not shards:
        raise FileNotFoundError(f"No prediction output for {path}")
    return pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True, copy=False)


def _archetype(frame: pd.DataFrame) -> pd.Series:
    for name in ("__archetype_policy_key__", "archetype_policy_key", "policy_archetype"):
        if name in frame.columns:
            return frame[name].astype(str)
    return pd.Series("unknown", index=frame.index)


def _top10(frame: pd.DataFrame, score: str) -> pd.Series:
    ordered = frame.loc[:, ["__ts__", "__symbol__", "side_name", score]].copy()
    ordered[score] = pd.to_numeric(ordered[score], errors="coerce")
    ordered["_symbol"] = ordered["__symbol__"].astype(str)
    ordered["_side"] = ordered["side_name"].astype(str)
    ordered = ordered.sort_values(["__ts__", score, "_symbol", "_side"], ascending=[True, False, True, True], kind="mergesort")
    ordered["_rank"] = ordered.groupby("__ts__", sort=False).cumcount()
    ordered["_n"] = ordered.groupby("__ts__", sort=False)[score].transform("size")
    selected_index = ordered.index[ordered["_rank"].lt(np.ceil(0.10 * ordered["_n"]))]
    return pd.Series(frame.index.isin(selected_index), index=frame.index)


def _metric(frame: pd.DataFrame, group: dict[str, object]) -> dict[str, object]:
    result = dict(group)
    result["selected_rows"] = int(len(frame))
    if frame.empty:
        return result
    numeric = lambda col: pd.to_numeric(frame.get(col), errors="coerce")
    result.update(
        mean_ev_after_1pct=float(numeric("ev_after_1pct").mean()),
        sum_ev_after_1pct=float(numeric("ev_after_1pct").sum()),
        mean_exec_margin=float(numeric("exec_margin").mean()),
        clean_exec_precision=float(numeric("clean_exec").mean()),
        first_touch_bad_mae_rate=float(numeric("first_touch_bad_mae_1r").mean()),
        full_path_bad_mae_rate=float(numeric("full_path_bad_mae_1r").mean()),
        timeout_rate=float(numeric("timeout").mean()),
        positive_ev_rate=float(numeric("ev_after_1pct").gt(0.0).mean()),
    )
    if "rapid_reversal" in frame:
        result["rapid_reversal_rate"] = float(numeric("rapid_reversal").mean())
    if "severe_retention" in frame:
        result["severe_retention_rate"] = float(numeric("severe_retention").mean())
    daily = frame.assign(_date=pd.to_datetime(frame["__ts__"], utc=True).dt.date).groupby("_date", sort=True).apply(
        lambda group: float((pd.to_numeric(group["clean_exec"], errors="coerce") - pd.to_numeric(group["score_meta_base_soft_label"], errors="coerce")).mean()),
        include_groups=False,
    )
    result["daily_signed_score_residual_autocorr_lag1"] = float(daily.autocorr(lag=1)) if len(daily) >= 3 else float("nan")
    return result


def _load_path_labels(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path)
        required = {*KEYS, "target", "target_realized", "model"}
        if not required.issubset(raw.columns):
            continue
        subset = raw.loc[
            raw["model"].eq("ebm") & raw["target"].isin(PATH_LABEL_TARGETS),
            [*KEYS, "target", "target_realized"],
        ]
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=[*KEYS, *PATH_LABEL_TARGETS])
    all_rows = pd.concat(rows, ignore_index=True, copy=False)
    if all_rows.duplicated([*KEYS, "target"]).any():
        raise ValueError("Path labels are not unique on the OOS candidate key.")
    return all_rows.pivot(index=list(KEYS), columns="target", values="target_realized").reset_index()


def _feature_usage(arm_dir: Path, arm: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_path in sorted((arm_dir / "models").glob("*/base_soft_label_*.joblib")):
        import joblib

        model = joblib.load(model_path)
        if not hasattr(model, "booster_"):
            continue
        booster = model.booster_
        names = list(booster.feature_name())
        split = booster.feature_importance(importance_type="split")
        gain = booster.feature_importance(importance_type="gain")
        dump = booster.dump_model()
        depths: dict[int, list[int]] = {}

        def visit(node: dict, depth: int) -> None:
            if "split_feature" not in node:
                return
            idx = int(node["split_feature"])
            depths.setdefault(idx, []).append(depth)
            visit(node["left_child"], depth + 1)
            visit(node["right_child"], depth + 1)

        for tree in dump.get("tree_info", []):
            visit(tree.get("tree_structure", {}), 0)
        for idx, feature in enumerate(names):
            if split[idx] <= 0:
                continue
            rows.append(
                {
                    "arm": arm,
                    "fold": model_path.parent.name,
                    "model": model_path.stem,
                    "feature": feature,
                    "split_count": int(split[idx]),
                    "gain": float(gain[idx]),
                    "mean_split_depth": float(np.mean(depths.get(idx, [np.nan]))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--path-predictions", type=Path, action="append", default=[])
    args = parser.parse_args()
    arm_dirs = [path for path in sorted(args.root.iterdir()) if path.is_dir() and (path / "manifest.json").exists()]
    if not arm_dirs:
        raise FileNotFoundError(f"No completed arm directories below {args.root}")
    path_inputs = list(args.path_predictions)
    if not path_inputs:
        path_inputs = sorted(Path("data_perp/reports").glob("breakout_path_quality_learnability_20260713_context_*/oof_predictions.parquet"))
    labels = _load_path_labels(path_inputs)
    metrics: list[dict[str, object]] = []
    churn: list[dict[str, object]] = []
    usage: list[pd.DataFrame] = []
    selections: dict[str, pd.DataFrame] = {}
    for arm_dir in arm_dirs:
        arm = arm_dir.name
        frame = _read_arm(arm_dir)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        frame["_archetype"] = _archetype(frame)
        frame = frame.merge(labels, on=list(KEYS), how="left", validate="many_to_one")
        frame["_selected_top10"] = _top10(frame, "score_meta_base_soft_label")
        selected = frame.loc[frame["_selected_top10"]].copy()
        selections[arm] = selected.loc[:, [*KEYS, "_archetype", "_selected_top10", "score_meta_base_soft_label"]]
        metrics.append(_metric(selected, {"arm": arm, "scope": "global_top10"}))
        target = selected.loc[selected["_archetype"].eq(TARGET_ARCHETYPE)]
        metrics.append(_metric(target, {"arm": arm, "scope": "short_breakout_top10"}))
        for month, group in target.groupby(pd.to_datetime(target["__ts__"], utc=True).dt.to_period("M").astype(str), sort=True):
            metrics.append(_metric(group, {"arm": arm, "scope": "short_breakout_top10_month", "month": month}))
        baseline_margin = frame.groupby("__ts__")["score_meta_base_soft_label"].transform(lambda values: values.quantile(0.90))
        target = target.assign(_margin=target["score_meta_base_soft_label"] - baseline_margin.loc[target.index])
        target["margin_bucket"] = pd.qcut(target["_margin"], q=min(4, target["_margin"].nunique()), duplicates="drop") if target["_margin"].nunique() > 1 else "flat"
        for bucket, group in target.groupby("margin_bucket", observed=False):
            metrics.append(_metric(group, {"arm": arm, "scope": "short_breakout_top10_margin", "margin_bucket": str(bucket)}))
        usage.append(_feature_usage(arm_dir, arm))
    if "baseline" in selections:
        base = selections["baseline"].rename(columns={"_selected_top10": "baseline_selected"})
        for arm, selected in selections.items():
            if arm == "baseline":
                continue
            joined = base.merge(selected.rename(columns={"_selected_top10": "arm_selected"}), on=[*KEYS, "_archetype"], how="outer")
            joined[["baseline_selected", "arm_selected"]] = joined[["baseline_selected", "arm_selected"]].fillna(False)
            affected = joined.loc[joined["_archetype"].eq(TARGET_ARCHETYPE)]
            churn.append({
                "arm": arm,
                "short_breakout_rows_added": int((~affected["baseline_selected"] & affected["arm_selected"]).sum()),
                "short_breakout_rows_removed": int((affected["baseline_selected"] & ~affected["arm_selected"]).sum()),
                "short_breakout_rows_unchanged": int((affected["baseline_selected"] & affected["arm_selected"]).sum()),
            })
    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty and "baseline" in set(metrics_df["arm"]):
        keys = [col for col in ("scope", "month", "margin_bucket") if col in metrics_df]
        for key in keys:
            metrics_df[key] = metrics_df[key].fillna("__all__").astype(str)
        numeric = [col for col in metrics_df.select_dtypes(include="number").columns if col != "selected_rows"]
        baseline = metrics_df.loc[metrics_df["arm"].eq("baseline"), [*keys, *numeric]].copy()
        baseline = baseline.rename(columns={col: f"_baseline_{col}" for col in numeric})
        metrics_df = metrics_df.merge(baseline, on=keys, how="left", validate="many_to_one")
        for col in numeric:
            metrics_df[f"delta_vs_baseline_{col}"] = metrics_df[col] - metrics_df[f"_baseline_{col}"]
        metrics_df = metrics_df.drop(columns=[f"_baseline_{col}" for col in numeric])
    metrics_df.to_csv(args.root / "metrics.csv", index=False)
    pd.DataFrame(churn).to_csv(args.root / "selection_churn_vs_baseline.csv", index=False)
    pd.concat([part for part in usage if not part.empty], ignore_index=True).to_csv(args.root / "feature_usage.csv", index=False) if any(not part.empty for part in usage) else pd.DataFrame().to_csv(args.root / "feature_usage.csv", index=False)
    (args.root / "report_manifest.json").write_text(json.dumps({
        "path_label_sources": [str(path) for path in path_inputs],
        "selection_contract": "identical timestamp-level top-10% budget per arm",
        "path_labels": "joined after OOS scoring for reporting only",
        "scope": {"side": "short", "archetype": TARGET_ARCHETYPE},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
