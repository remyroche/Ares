#!/usr/bin/env python3
"""S52 cost-ladder report over OOF scores and score blends.

The S52 label plan calls for evaluating 0/10/25/50/100 bps after path ordering,
not before. This script reuses already OOF scored rows, derives gross edge from
the ledger's original-cost net columns, then recomputes first-touch and utility
net metrics at each requested cost level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_score_blend_ablation import (  # noqa: E402
    DEFAULT_LEDGER,
    _blend_scores,
    _evaluate_score,
    _json_safe,
    _parse_weights,
    _wide_scores,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/s52_cost_ladder_learnability_features_noae_20260705_v1"
)
DEFAULT_COSTS_BPS = "0,10,25,50,100"
NET_COLUMNS = ("first_touch_net", "u_policy_net", "ret_net")


def _parse_costs_bps(raw: str) -> list[float]:
    costs: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        bps = float(token)
        if bps < 0.0:
            raise ValueError(f"cost bps must be non-negative: {bps}")
        costs.append(bps / 10000.0)
    return sorted(set(costs))


def _adjust_cost(base: pd.DataFrame, *, original_cost: float, new_cost: float) -> pd.DataFrame:
    adjusted = base.copy()
    delta = float(original_cost) - float(new_cost)
    for col in NET_COLUMNS:
        if col in adjusted.columns:
            adjusted[col] = pd.to_numeric(adjusted[col], errors="coerce") + delta
    return adjusted


def _best_variants(summary: pd.DataFrame, *, max_variants: int, selection_metric: str = "objective") -> list[str]:
    if summary.empty or "variant" not in summary.columns:
        return []
    metric = str(selection_metric or "objective")
    if metric not in summary.columns:
        raise ValueError(f"selection metric {metric!r} not found in summary columns")
    ranked = summary.sort_values(metric, ascending=False).reset_index(drop=True)
    variants = [str(v) for v in ranked["variant"].head(int(max_variants)).tolist()]
    singles = [str(v) for v in summary["variant"].tolist() if str(v).startswith("single::")]
    return list(dict.fromkeys(variants + singles))


def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 40) -> str:
        if df.empty:
            return "No rows."
        view = df[[col for col in cols if col in df.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    top_cols = [
        "cost_bps",
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_timeout_rate",
        "mean_long_top10_mean_first_touch_net",
        "mean_short_top10_mean_first_touch_net",
    ]
    per_cost = (
        summary.sort_values(["cost_bps", "objective"], ascending=[True, False])
        .groupby("cost_bps", observed=True, dropna=False)
        .head(5)
        .reset_index(drop=True)
    )
    lines = [
        "# S52 Cost Ladder",
        "",
        "This is an OOF diagnostic. It adjusts net edge from the ledger's original-cost net columns and keeps the OOF score ordering fixed.",
        "",
        f"Ledger: `{manifest['ledger']}`",
        f"Original cost: `{manifest['original_cost_bps']:.1f}` bps",
        f"Costs: `{', '.join(str(c) for c in manifest['costs_bps'])}` bps",
        f"Normalization: `{manifest['normalization']}`",
        "",
        "## Best Rows By Cost",
        "",
        fmt(per_cost, top_cols, n=100),
        "",
        "## Best Overall",
        "",
        fmt(summary.sort_values("objective", ascending=False), top_cols, n=30),
        "",
    ]
    output_dir.joinpath("s52_cost_ladder.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    ledger_path: Path,
    output_dir: Path,
    original_cost: float,
    costs: list[float],
    weights: list[float],
    normalization: str,
    max_variants_per_cost: int,
    selection_metric: str = "objective",
) -> None:
    ledger = pd.read_parquet(ledger_path)
    base, score_frame = _wide_scores(ledger)
    blends = _blend_scores(base, score_frame, weights=weights, normalization=normalization)

    pre_summaries: list[dict[str, Any]] = []
    for name, score in blends.items():
        summary, _rows = _evaluate_score(base, name, score, round_trip_cost=float(original_cost))
        pre_summaries.append(summary)
    selected = set(
        _best_variants(
            pd.DataFrame(pre_summaries),
            max_variants=max_variants_per_cost,
            selection_metric=str(selection_metric),
        )
    )

    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for cost in costs:
        adjusted = _adjust_cost(base, original_cost=float(original_cost), new_cost=float(cost))
        for name, score in blends.items():
            if name not in selected:
                continue
            summary, rows = _evaluate_score(adjusted, name, score, round_trip_cost=float(cost))
            cost_bps = float(cost) * 10000.0
            summary["cost_bps"] = cost_bps
            summary["original_cost_bps"] = float(original_cost) * 10000.0
            summary["normalization_scope"] = str(normalization)
            summaries.append(summary)
            for row in rows:
                row["cost_bps"] = cost_bps
                row["original_cost_bps"] = float(original_cost) * 10000.0
                fold_rows.append(row)

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values(["cost_bps", "objective"], ascending=[True, False])
        .reset_index(drop=True)
    )
    folds_df = pd.DataFrame(fold_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "s52_cost_ladder_summary.csv"
    folds_path = output_dir / "s52_cost_ladder_folds.csv"
    manifest_path = output_dir / "manifest.json"
    summary_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)
    manifest = {
        "ledger": str(ledger_path),
        "output_dir": str(output_dir),
        "rows": int(len(base)),
        "original_cost_bps": float(original_cost) * 10000.0,
        "costs_bps": [float(cost) * 10000.0 for cost in costs],
        "normalization": str(normalization),
        "selection_metric": str(selection_metric),
        "weights": [float(w) for w in weights],
        "selected_variants": sorted(selected),
        "outputs": {
            "summary": str(summary_path),
            "folds": str(folds_path),
            "report": str(output_dir / "s52_cost_ladder.md"),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir, summary_df, folds_df, manifest)
    print(f"wrote {summary_path}")
    cols = [
        "cost_bps",
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_timeout_rate",
    ]
    print(summary_df[cols].groupby("cost_bps", observed=True, dropna=False).head(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--original-cost-bps", type=float, default=100.0)
    parser.add_argument("--costs-bps", default=DEFAULT_COSTS_BPS)
    parser.add_argument("--weights", default="0,0.25,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument(
        "--normalization",
        choices=("global", "month", "month_side", "timestamp_side"),
        default="global",
    )
    parser.add_argument("--max-variants-per-cost", type=int, default=12)
    parser.add_argument(
        "--selection-metric",
        default="objective",
        help="Metric used to shortlist variants before recomputing the cost ladder.",
    )
    args = parser.parse_args()
    run(
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        original_cost=float(args.original_cost_bps) / 10000.0,
        costs=_parse_costs_bps(args.costs_bps),
        weights=_parse_weights(args.weights),
        normalization=str(args.normalization),
        max_variants_per_cost=int(args.max_variants_per_cost),
        selection_metric=str(args.selection_metric),
    )


if __name__ == "__main__":
    main()
