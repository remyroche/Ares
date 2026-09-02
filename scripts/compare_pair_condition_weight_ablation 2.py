#!/usr/bin/env python3
"""Compare the predeclared soft-membership weighting exponents.

The three runs share discovery artifacts, candidate IDs, folds, model
parameters and score-calibration contracts.  This utility only joins their
already materialized OOS outputs; it never reselects an exponent using OOS
labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "data_perp/artifacts"
RUNS = {
    1.0: ARTIFACT_ROOT / "pair_condition_specialists_20260806_gamma1p0",
    1.5: ARTIFACT_ROOT / "pair_condition_specialists_20260806_v5",
    2.0: ARTIFACT_ROOT / "pair_condition_specialists_20260806_gamma2p0",
}
OUT = ARTIFACT_ROOT / "pair_condition_specialists_weight_ablation_20260806"


def _metric_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exponent, path in RUNS.items():
        metrics = pd.read_parquet(path / "global_metrics.parquet")
        pooled = metrics[(metrics.scope == "global") & (metrics.period == "all") & metrics["tail"].isin([.01, .05, .10])]
        for rec in pooled.to_dict("records"):
            rows.append({"exponent": exponent, "evaluation": "h12_global", **rec})
        monthly = metrics[(metrics.scope == "global") & metrics.period.astype(str).str.match(r"^2024-") & metrics["tail"].eq(.05)]
        for rec in monthly.to_dict("records"):
            rows.append({"exponent": exponent, "evaluation": "h12_monthly_top5", **rec})
        exit_metrics = pd.read_parquet(path / "fixed_exit_metrics.parquet")
        for rec in exit_metrics.to_dict("records"):
            rows.append({"exponent": exponent, "evaluation": "fixed_exit_global", "system": rec["system"], "tail": np.nan, **rec})
        lomo = pd.read_parquet(path / "condition_lomo_results.parquet")
        if not lomo.empty and "held_top10_delta_net_bps" in lomo:
            for side, g in lomo.groupby("side"):
                values = g.held_top10_delta_net_bps.to_numpy(float)
                rows.append({
                    "exponent": exponent,
                    "evaluation": "lomo_top10_delta",
                    "system": f"lomo:{side}",
                    "side": side,
                    "rows": len(g),
                    "net_bps": float(np.median(values)),
                    "min_net_bps": float(np.min(values)),
                    "max_net_bps": float(np.max(values)),
                    "positive_fraction": float(np.mean(values > 0.0)),
                })
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    h = metrics[(metrics.evaluation == "h12_global") & metrics.system.isin(["anchor_only", "raw_ranks", "full_context", "gated_ranks", "memberships", "innovations", "gated_innovations"]) & metrics["tail"].isin([.01, .05, .10])].copy()
    wide = h.pivot_table(index=["exponent", "system"], columns="tail", values="net_bps", aggfunc="first").reset_index()
    return wide.rename(columns={.01: "top1_net_bps", .05: "top5_net_bps", .10: "top10_net_bps"})


def _markdown_table(frame: pd.DataFrame) -> str:
    x = frame.copy()
    headers = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in x.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                cells.append(f"{float(value):.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _mad(values: pd.Series) -> float:
    x = np.asarray(values, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def _positive_fraction(values: pd.Series) -> float:
    return float(np.mean(np.asarray(values, dtype=float) > 0.0))


def run() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = _metric_rows()
    metrics.to_parquet(OUT / "condition_weight_ablation.parquet", index=False)
    summary = _summary(metrics)
    summary.to_csv(OUT / "condition_weight_ablation_summary.csv", index=False)
    monthly = metrics[metrics.evaluation.eq("h12_monthly_top5")].copy()
    monthly_summary = (
        monthly.groupby(["exponent", "system"], as_index=False)
        .agg(
            months=("period", "nunique"),
            median_month_top5_net_bps=("net_bps", "median"),
            worst_month_top5_net_bps=("net_bps", "min"),
            best_month_top5_net_bps=("net_bps", "max"),
            month_mad_top5_net_bps=("net_bps", _mad),
            positive_month_fraction=("net_bps", _positive_fraction),
        )
    )
    monthly_summary.to_parquet(OUT / "condition_weight_monthly_summary.parquet", index=False)
    anchor = summary[summary.system.eq("anchor_only")].set_index("exponent")
    lines = [
        "# Condition-membership weighting ablation",
        "",
        "The three arms use the same frozen discovery conditions, feature sets, OOS folds, residual target, side-local EV maps and global ranking contract. Only the membership/gating exponent changes: γ ∈ {1.0, 1.5, 2.0}.",
        "",
        "## H12 global net bps/trade",
        "",
        _markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "The no-specialist anchor is invariant because it does not consume specialist weights. Promotion requires the specialist arm to improve both top-5 and top-10 net versus the matched anchor and not worsen the worst transport month. The comparison table is descriptive; no exponent was selected from final OOS outcomes.",
        "",
        "## Monthly stability",
        "",
        _markdown_table(monthly_summary[monthly_summary.system.isin(["anchor_only", "raw_ranks", "full_context"]) ]),
        "",
        "## LOMO",
        "",
        "metrics with evaluation=lomo_top10_delta report the true train-only threshold/model/residualizer portability audit. Positive fraction is the share of held-out discovery months where the specialist beat the base-EV anchor at top-10.",
        "",
        "Artifacts: `condition_weight_ablation.parquet`, `condition_weight_ablation_summary.csv`, `condition_weight_monthly_summary.parquet`.",
    ]
    (OUT / "CONDITION_WEIGHT_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "schema": "pair_condition_weight_ablation_v1",
        "exponents": [1.0, 1.5, 2.0],
        "runs": {str(k): str(v) for k, v in RUNS.items()},
        "shared_contract": "frozen discovery / residual target / side-local mapping / global common-bps top-k",
        "selection": "descriptive; no final OOS exponent selection",
        "metrics_artifact": "condition_weight_ablation.parquet",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return OUT


if __name__ == "__main__":
    print(run())
