#!/usr/bin/env python3
"""Chronological month-forward validation for contextual TP/SL overlay variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _variant_dirs(root: Path, exclude_parts: set[str]) -> List[Path]:
    out: List[Path] = []
    for path in root.glob("*/materialized/*"):
        if any(part in exclude_parts for part in path.parts):
            continue
        if (path / "combo_replay_manifest.json").exists():
            out.append(path)
    return sorted(out)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads((path / "combo_replay_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _global_day_rows(path: Path) -> pd.DataFrame:
    frame = _read_csv(path / "combo_replay_daily_metrics.csv")
    if frame.empty:
        return pd.DataFrame()
    out = frame.loc[frame["period_type"].astype(str).eq("day")].copy()
    if "head" in out.columns:
        out = out.loc[out["head"].isna()].copy()
    out["day"] = pd.to_datetime(out["day"], errors="coerce")
    out["month"] = out["day"].dt.to_period("M").astype(str)
    return out


def _monthly_metrics(days: pd.DataFrame, *, q35_weight: float, q20_weight: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for month, group in days.groupby("month", sort=True):
        pnl = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
        if pnl.empty:
            continue
        rows.append(
            {
                "month": str(month),
                "net_pnl": float(pnl.sum()),
                "avg_day_net_pnl": float(pnl.mean()),
                "q20_day_net_pnl": float(pnl.quantile(0.20)),
                "q35_day_net_pnl": float(pnl.quantile(0.35)),
                "q50_day_net_pnl": float(pnl.quantile(0.50)),
                "day_count": int(len(pnl)),
                "trade_count": int(pd.to_numeric(group.get("trades", pd.Series(dtype=float)), errors="coerce").sum()),
                "hit_rate": float(pd.to_numeric(group.get("hit_rate", pd.Series(dtype=float)), errors="coerce").mean()),
                "full_sl_rate": float(pd.to_numeric(group.get("full_sl_rate", pd.Series(dtype=float)), errors="coerce").mean()),
                "timeout_rate": float(pd.to_numeric(group.get("timeout_rate", pd.Series(dtype=float)), errors="coerce").mean()),
                "month_objective": float(
                    pnl.mean() + q35_weight * pnl.quantile(0.35) + q20_weight * pnl.quantile(0.20)
                ),
            }
        )
    return pd.DataFrame(rows)


def _score_training(history: pd.DataFrame) -> Dict[str, float]:
    return {
        "train_months": int(history["month"].nunique()),
        "train_sum_net_pnl": float(history["net_pnl"].sum()),
        "train_avg_month_objective": float(history["month_objective"].mean()),
        "train_q25_month_objective": float(history["month_objective"].quantile(0.25)),
        "train_positive_month_share": float((history["net_pnl"] > 0.0).mean()),
    }


def _selector_score(row: pd.Series, positive_month_weight: float) -> float:
    return float(
        row["train_avg_month_objective"]
        + 0.35 * row["train_q25_month_objective"]
        + positive_month_weight * row["train_positive_month_share"] * max(abs(row["train_avg_month_objective"]), 1.0)
    )


def _markdown_table(frame: pd.DataFrame, columns: List[str], limit: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    cur = frame[[col for col in columns if col in frame.columns]].head(limit).copy()
    for col in cur.columns:
        if pd.api.types.is_float_dtype(cur[col]):
            cur[col] = cur[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return cur.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", action="append", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--positive-month-weight", type=float, default=0.10)
    parser.add_argument(
        "--exclude-root-names",
        default="noop_baseline,summary,tail_tradeoff_summary,grid_decision_summary,interaction_summary",
    )
    args = parser.parse_args()

    roots = [Path(path) for path in args.root_dir]
    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_parts = {part.strip() for part in str(args.exclude_root_names).split(",") if part.strip()}

    baseline_month = _monthly_metrics(
        _global_day_rows(baseline_dir),
        q35_weight=float(args.q35_weight),
        q20_weight=float(args.q20_weight),
    )
    baseline_month = baseline_month.add_prefix("baseline_").rename(columns={"baseline_month": "month"})

    variant_months: List[pd.DataFrame] = []
    for root in roots:
        for path in _variant_dirs(root, exclude_parts):
            manifest = _load_manifest(path)
            days = _global_day_rows(path)
            if days.empty:
                continue
            metrics = _monthly_metrics(days, q35_weight=float(args.q35_weight), q20_weight=float(args.q20_weight))
            metrics.insert(0, "label", path.name)
            metrics.insert(0, "source_root", str(root))
            metrics["rule_count"] = len(manifest.get("rules", []))
            metrics["net_pnl_total"] = manifest.get("metrics", {}).get("net_pnl")
            variant_months.append(metrics)
    all_months = pd.concat(variant_months, ignore_index=True) if variant_months else pd.DataFrame()
    if all_months.empty:
        raise ValueError("No variant monthly metrics found")
    all_months = all_months.merge(baseline_month, on="month", how="left")
    for col in ["net_pnl", "month_objective", "q20_day_net_pnl", "q35_day_net_pnl", "avg_day_net_pnl"]:
        all_months[f"delta_{col}"] = pd.to_numeric(all_months[col], errors="coerce") - pd.to_numeric(
            all_months[f"baseline_{col}"], errors="coerce"
        )

    months = sorted(all_months["month"].unique())
    selections: List[Dict[str, Any]] = []
    rankings: List[pd.DataFrame] = []
    for eval_month in months[int(args.min_train_months) :]:
        history = all_months.loc[all_months["month"].lt(eval_month)].copy()
        if history["month"].nunique() < int(args.min_train_months):
            continue
        eval_rows = all_months.loc[all_months["month"].eq(eval_month)].copy()
        train_rows: List[Dict[str, Any]] = []
        for (label, source_root), group in history.groupby(["label", "source_root"], sort=False):
            stats = _score_training(group)
            stats.update({"label": label, "source_root": source_root, "eval_month": eval_month})
            train_rows.append(stats)
        rank = pd.DataFrame(train_rows)
        rank["selector_score"] = rank.apply(lambda row: _selector_score(row, float(args.positive_month_weight)), axis=1)
        rank = rank.sort_values(["selector_score", "train_sum_net_pnl"], ascending=False)
        rank["rank"] = np.arange(1, len(rank) + 1)
        rankings.append(rank)
        chosen = rank.iloc[0]
        chosen_eval = eval_rows.loc[
            eval_rows["label"].eq(chosen["label"]) & eval_rows["source_root"].eq(chosen["source_root"])
        ]
        if chosen_eval.empty:
            continue
        erow = chosen_eval.iloc[0].to_dict()
        selections.append(
            {
                "eval_month": eval_month,
                "chosen_label": chosen["label"],
                "chosen_source_root": chosen["source_root"],
                "selector_score": float(chosen["selector_score"]),
                "train_months": int(chosen["train_months"]),
                "train_sum_net_pnl": float(chosen["train_sum_net_pnl"]),
                "train_avg_month_objective": float(chosen["train_avg_month_objective"]),
                "train_q25_month_objective": float(chosen["train_q25_month_objective"]),
                "train_positive_month_share": float(chosen["train_positive_month_share"]),
                "eval_net_pnl": float(erow["net_pnl"]),
                "baseline_eval_net_pnl": float(erow["baseline_net_pnl"]),
                "delta_eval_net_pnl": float(erow["delta_net_pnl"]),
                "eval_month_objective": float(erow["month_objective"]),
                "baseline_eval_month_objective": float(erow["baseline_month_objective"]),
                "delta_eval_month_objective": float(erow["delta_month_objective"]),
                "delta_eval_q20_day_net_pnl": float(erow["delta_q20_day_net_pnl"]),
                "delta_eval_q35_day_net_pnl": float(erow["delta_q35_day_net_pnl"]),
                "rule_count": int(erow.get("rule_count", 0)),
            }
        )

    selections_df = pd.DataFrame(selections)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    all_months.to_csv(out_dir / "month_forward_all_variant_months.csv", index=False)
    rankings_df.to_csv(out_dir / "month_forward_training_rankings.csv", index=False)
    selections_df.to_csv(out_dir / "month_forward_selections.csv", index=False)

    if not selections_df.empty:
        summary = {
            "eval_months": int(len(selections_df)),
            "positive_net_months": int((selections_df["delta_eval_net_pnl"] > 0.0).sum()),
            "positive_objective_months": int((selections_df["delta_eval_month_objective"] > 0.0).sum()),
            "sum_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].sum()),
            "mean_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].mean()),
            "mean_delta_month_objective": float(selections_df["delta_eval_month_objective"].mean()),
            "q25_delta_month_objective": float(selections_df["delta_eval_month_objective"].quantile(0.25)),
            "worst_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].min()),
            "chosen_labels": selections_df["chosen_label"].value_counts().to_dict(),
        }
    else:
        summary = {"eval_months": 0}
    (out_dir / "month_forward_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Contextual TP/SL Overlay Month-Forward Validation",
        "",
        "Candidate selection uses only months before the evaluation month.",
        f"Objective: avg_day_net_pnl + {args.q35_weight:g} * q35_day_net_pnl + {args.q20_weight:g} * q20_day_net_pnl.",
        "",
        "## Summary",
        "",
        _markdown_table(pd.DataFrame([summary]), list(summary.keys())),
        "",
        "## Month-Forward Selections",
        "",
        _markdown_table(
            selections_df,
            [
                "eval_month",
                "chosen_label",
                "train_months",
                "train_positive_month_share",
                "delta_eval_net_pnl",
                "delta_eval_month_objective",
                "delta_eval_q20_day_net_pnl",
                "delta_eval_q35_day_net_pnl",
                "rule_count",
            ],
            40,
        ),
    ]
    (out_dir / "month_forward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "generated_by": "validate_contextual_tp_sl_overlay_month_forward",
        "root_dirs": [str(root) for root in roots],
        "baseline_dir": str(baseline_dir),
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "min_train_months": int(args.min_train_months),
        "positive_month_weight": float(args.positive_month_weight),
        "outputs": [
            "month_forward_report.md",
            "month_forward_summary.json",
            "month_forward_all_variant_months.csv",
            "month_forward_training_rankings.csv",
            "month_forward_selections.csv",
        ],
    }
    (out_dir / "month_forward_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
