#!/usr/bin/env python3
"""Diagnose pre-trade features behind size-action defensive-success gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXCLUDE_COLUMNS = {
    "timestamp",
    "strategy_id",
    "head",
    "component_scope",
    "accepted",
    "reject_reason",
    "head_specific_component",
    "scorer_intervention",
    "fold_week_start",
    "baseline_group_net_pnl",
    "baseline_group_winner_pnl",
    "baseline_group_loser_loss",
    "baseline_group_trades",
    "defensive_success_value",
    "defensive_success_target",
    "gate_probability",
    "gate_threshold",
    "gate_keep",
}


def _week_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")


def _rank_auc(feature: pd.Series, target: pd.Series) -> float:
    x = pd.to_numeric(feature, errors="coerce")
    y = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    valid = x.notna() & y.isin([0, 1])
    x = x.loc[valid]
    y = y.loc[valid]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = x.rank(method="average")
    rank_sum_pos = float(ranks.loc[y.eq(1)].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 8 or values.nunique(dropna=True) < 2:
            continue
        cols.append(col)
    return cols


def _feature_diagnostics(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = pd.to_numeric(frame["defensive_success_target"], errors="coerce").fillna(0).astype(int)
    value = pd.to_numeric(frame["defensive_success_value"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = x.notna()
        if valid.sum() < 8:
            continue
        pos = x.loc[valid & y.eq(1)]
        neg = x.loc[valid & y.eq(0)]
        pooled_std = float(x.loc[valid].std(ddof=0))
        pos_mean = float(pos.mean()) if len(pos) else float("nan")
        neg_mean = float(neg.mean()) if len(neg) else float("nan")
        mean_delta = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else float("nan")
        rows.append(
            {
                "feature": col,
                "non_null_rows": int(valid.sum()),
                "unique_values": int(x.nunique(dropna=True)),
                "auc_success": _rank_auc(x, y),
                "auc_success_oriented": float(max(_rank_auc(x, y), 1.0 - _rank_auc(x, y)))
                if np.isfinite(_rank_auc(x, y))
                else float("nan"),
                "spearman_value": float(x.corr(value, method="spearman")) if valid.sum() >= 8 else float("nan"),
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "mean_delta_pos_minus_neg": mean_delta,
                "standardized_delta": float(mean_delta / pooled_std) if pooled_std > 0 and np.isfinite(mean_delta) else float("nan"),
                "missing_rate": float(1.0 - valid.mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = out[["auc_success_oriented", "spearman_value", "standardized_delta"]].assign(
        spearman_value=lambda d: d["spearman_value"].abs(),
        standardized_delta=lambda d: d["standardized_delta"].abs(),
    ).fillna(0.0).sum(axis=1)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def _write_markdown(
    path: Path,
    *,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    top_global: pd.DataFrame,
    top_by_head: dict[str, pd.DataFrame],
) -> None:
    lines: list[str] = ["# Size-action gate feature diagnostics", ""]
    lines.append("## Manifest")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Intervention Outcomes")
    lines.append("")
    lines.append(aggregate.to_markdown(index=False) if not aggregate.empty else "_No interventions._")
    lines.append("")
    lines.append("## Top Global Feature Separators")
    lines.append("")
    keep_cols = [
        "feature",
        "auc_success",
        "spearman_value",
        "standardized_delta",
        "pos_mean",
        "neg_mean",
    ]
    lines.append(top_global[keep_cols].head(25).to_markdown(index=False) if not top_global.empty else "_No features._")
    for head, frame in top_by_head.items():
        lines.append("")
        lines.append(f"## Top Feature Separators: {head}")
        lines.append("")
        lines.append(frame[keep_cols].head(20).to_markdown(index=False) if not frame.empty else "_No features._")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-frame", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-head-rows", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.training_frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["week_start"] = _week_start(frame["timestamp"])
    frame["scorer_intervention"] = frame.get("scorer_intervention", False).astype(bool)
    interventions = frame.loc[frame["scorer_intervention"]].copy()
    interventions["defensive_success_value"] = pd.to_numeric(interventions["defensive_success_value"], errors="coerce").fillna(0.0)
    interventions["defensive_success_target"] = pd.to_numeric(
        interventions["defensive_success_target"], errors="coerce"
    ).fillna(0).astype(int)

    aggregate = interventions.groupby(["week_start", "head"], dropna=False).agg(
        intervention_rows=("strategy_id", "size"),
        positive_rate=("defensive_success_target", "mean"),
        defensive_success_sum=("defensive_success_value", "sum"),
        defensive_success_mean=("defensive_success_value", "mean"),
        selected_multiplier_mean=("selected_multiplier", "mean"),
        p_intervene_mean=("p_intervene", "mean"),
        pred_delta_J_mean=("pred_delta_J", "mean"),
    ).reset_index()
    aggregate.to_csv(args.out_dir / "intervention_outcomes_by_week_head.csv", index=False)

    feature_cols = _numeric_feature_columns(interventions)
    global_diag = _feature_diagnostics(interventions, feature_cols)
    global_diag.to_csv(args.out_dir / "feature_diagnostics_global.csv", index=False)

    top_by_head: dict[str, pd.DataFrame] = {}
    for head, group in interventions.groupby("head", dropna=False):
        if len(group) < int(args.min_head_rows) or group["defensive_success_target"].nunique() < 2:
            continue
        diag = _feature_diagnostics(group, feature_cols)
        diag.to_csv(args.out_dir / f"feature_diagnostics_{head}.csv", index=False)
        top_by_head[str(head)] = diag

    manifest = {
        "generated_by": "diagnose_size_action_gate_features",
        "training_frame": str(args.training_frame),
        "rows": int(len(frame)),
        "intervention_rows": int(len(interventions)),
        "feature_count": int(len(feature_cols)),
        "heads": sorted(str(x) for x in interventions["head"].dropna().unique()),
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    _write_markdown(
        args.out_dir / "size_action_gate_feature_diagnostics.md",
        manifest=manifest,
        aggregate=aggregate,
        top_global=global_diag,
        top_by_head=top_by_head,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
