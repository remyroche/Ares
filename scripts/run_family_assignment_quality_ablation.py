#!/usr/bin/env python3
"""Evaluate predeclared trust gating for the semantic family correction layer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
QUALITY_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50)


def _metrics(frame: pd.DataFrame, score: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def one(block: pd.DataFrame, period: str) -> None:
        ordered = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            n = max(1, int(math.ceil(len(ordered) * tail)))
            chosen = ordered.head(n)
            net = chosen["policy_net_bps"].to_numpy(float)
            rows.append(
                {
                    "score": score,
                    "period": period,
                    "tail": tail,
                    "trades": int(n),
                    "gross_bps_per_trade": float(np.nanmean(chosen["policy_gross_bps"])),
                    "net_bps_per_trade": float(np.nanmean(net)),
                    "win_rate_net": float(np.mean(net > 0.0)),
                    "median_net_bps": float(np.nanmedian(net)),
                    "p10_net_bps": float(np.nanpercentile(net, 10)),
                }
            )

    one(frame, "pooled")
    month = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    for name, block in frame.assign(_period=month).groupby("_period", observed=True):
        one(block, str(name))
    week = pd.to_datetime(frame["__ts__"], utc=True).dt.to_period("W").astype(str)
    for name, block in frame.assign(_period=week).groupby("_period", observed=True):
        one(block, str(name))
    return rows


def run(args: argparse.Namespace) -> Path:
    source = Path(args.predictions)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    required = {
        "fold", "candidate_id", "__ts__", "split", "policy_net_bps", "policy_gross_bps",
        "cap120_policy_correction", "mlp_residual_delta", "mlp_state_confidence",
        "family_assignment_quality", "family_low_confidence_mass", "family_selected_mass",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction artifact missing required fields: {missing}")
    test = frame.loc[frame["split"].eq("test")].copy()
    if test.empty:
        raise ValueError("no outer-test rows")

    base = test["cap120_policy_correction"].to_numpy(float)
    delta = np.clip(0.50 * test["mlp_residual_delta"].to_numpy(float), -75.0, 75.0)
    quality = test["family_assignment_quality"].to_numpy(float).clip(0.0, 1.0)
    represented = test["family_selected_mass"].to_numpy(float).clip(0.0, 1.0)
    confidence = test["mlp_state_confidence"].to_numpy(float).clip(0.0, 1.0)

    scores: dict[str, np.ndarray] = {"arm_A_cap120": base}
    # Continuous attenuation uses only decision-time trust fields and a fixed
    # 0.45 reference; it is not tuned on the held-out outcomes.
    quality_scale = np.minimum(quality / 0.45, 1.0)
    scores["quality_weighted"] = base + delta * quality_scale
    mass_scale = np.minimum(represented / 0.80, 1.0)
    scores["quality_mass_weighted"] = base + delta * quality_scale * mass_scale
    for threshold in QUALITY_THRESHOLDS:
        q = f"{threshold:.2f}"
        scores[f"quality_gate_{q}"] = base + np.where(quality >= threshold, delta, 0.0)
        scores[f"trust_gate_{q}"] = base + np.where(
            (quality >= threshold) & (represented >= 0.80) & (confidence >= 0.45), delta, 0.0
        )

    metric_rows: list[dict[str, object]] = []
    for name, value in scores.items():
        test[name] = value.astype("float32")
        metric_rows.extend(_metrics(test, name))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(out / "quality_ablation_metrics.parquet", index=False, compression="zstd")
    month_metrics = metrics.loc[
        metrics["tail"].eq(0.05)
        & metrics["period"].astype(str).str.fullmatch(r"\d{4}-\d{2}")
    ].copy()
    monthly_rows = []
    for score, block in month_metrics.groupby("score", observed=True):
        worst = block.loc[block["net_bps_per_trade"].idxmin()]
        monthly_rows.append(
            {
                "score": score,
                "months": int(len(block)),
                "mean_net_bps_per_trade": float(block["net_bps_per_trade"].mean()),
                "median_net_bps_per_trade": float(block["net_bps_per_trade"].median()),
                "worst_net_bps_per_trade": float(worst["net_bps_per_trade"]),
                "worst_month": str(worst["period"]),
                "positive_months": int((block["net_bps_per_trade"] > 0.0).sum()),
            }
        )
    monthly = pd.DataFrame(monthly_rows).sort_values("mean_net_bps_per_trade", ascending=False)
    monthly.to_parquet(out / "quality_ablation_monthly_top5.parquet", index=False, compression="zstd")

    coverage = []
    for fold, block in test.groupby("fold", observed=True):
        for name, value in scores.items():
            mask = np.isfinite(value) & (value != base)
            idx = block.index.to_numpy()
            local = mask[np.searchsorted(test.index.to_numpy(), idx)] if test.index.is_monotonic_increasing else np.asarray(block[name].to_numpy(float) != block["cap120_policy_correction"].to_numpy(float))
            coverage.append(
                {
                    "fold": str(fold),
                    "score": name,
                    "rows": int(len(block)),
                    "correction_active_rate": float(np.mean(local)),
                    "quality_mean": float(block["family_assignment_quality"].mean()),
                    "represented_mass_mean": float(block["family_selected_mass"].mean()),
                    "low_confidence_mass_mean": float(block["family_low_confidence_mass"].mean()),
                }
            )
    pd.DataFrame(coverage).to_parquet(out / "quality_ablation_coverage.parquet", index=False, compression="zstd")

    pooled = metrics.loc[metrics["period"].eq("pooled") & metrics["tail"].isin([0.005, 0.01, 0.05, 0.10])]
    lines = [
        "# Family assignment-quality admission ablation",
        "",
        "This is a post-fit, strict outer-test diagnostic on the frozen semantic 80-family contract. All gates and attenuation constants were predeclared; no OOS outcome was used to choose a threshold.",
        "",
        "## Frozen trust rules",
        "",
        "- quality-weighted: correction × min(assignment_quality / 0.45, 1)",
        "- quality-mass-weighted: additionally × min(represented_mass / 0.80, 1)",
        "- quality gates: quality >= 0.30/0.35/0.40/0.45/0.50",
        "- trust gates: quality threshold, represented mass >= 0.80, and MLP confidence >= 0.45",
        "",
        "## Pooled global tails",
        "",
        "| score | tail | trades | gross | net | win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.sort_values(["tail", "net_bps_per_trade"], ascending=[True, False]).itertuples(index=False):
        lines.append(f"| {row.score} | {row.tail:.3g} | {row.trades} | {row.gross_bps_per_trade:.2f} | {row.net_bps_per_trade:.2f} | {row.win_rate_net:.3f} |")
    lines += [
        "",
        "## Monthly top-5 stability",
        "",
        "| score | mean net | median net | worst month | worst net | positive months |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in monthly.itertuples(index=False):
        lines.append(f"| {row.score} | {row.mean_net_bps_per_trade:.2f} | {row.median_net_bps_per_trade:.2f} | {row.worst_month} | {row.worst_net_bps_per_trade:.2f} | {row.positive_months}/{row.months} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The contract passes the >=80% represented-mass gate, but its low-confidence fallback mass is a separate trust risk. Promotion requires improvement over the Cap-120 control without a worst-month failure; this artifact is diagnostic and does not change the frozen production score.",
    ]
    (out / "QUALITY_ADMISSION_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "family_assignment_quality_ablation_v1",
        "status": "complete",
        "predictions": str(source),
        "outer_test_rows": int(len(test)),
        "thresholds": list(QUALITY_THRESHOLDS),
        "rules_predeclared": True,
        "oos_outcomes_used_for_selection": False,
        "scores": list(scores),
        "policy": "global pooled tails; exact 15m trailing policy labels already frozen upstream",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    print(run(_parser().parse_args()))
