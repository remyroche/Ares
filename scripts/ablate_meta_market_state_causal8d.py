#!/usr/bin/env python3
"""Leakage-safe 8-day smoother ablation over the market-state MLP champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_meta_market_state_threshold_calibration import (
    _causal_8d_residual_overlay,
    _score_metrics,
    _top10_mask,
)


POLICY_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
BLACKLIST = "long_dirtyavoid_sparse_questionable"


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).to_numpy(np.float32)


def _objective(metric: dict[str, float], baseline: dict[str, float]) -> float:
    gain = float(metric["mean_ev_after_1pct"] - baseline["mean_ev_after_1pct"])
    allowance = max(0.0, gain / 5.0)
    if metric["worst_week_ev"] < baseline["worst_week_ev"] - allowance:
        return -1e9
    if metric["worst_month_ev"] < baseline["worst_month_ev"] - allowance:
        return -1e9
    return float(
        metric["mean_ev_after_1pct"]
        + 0.20 * metric["worst_week_ev"]
        + 0.10 * metric["worst_month_ev"]
        - 0.10 * abs(metric.get("mean_negative_surprise", 0.0))
    )


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = frame.sort_values("__ts__", kind="stable").reset_index(drop=True)
    rank_col = (
        "expected_ev_rank_score"
        if "expected_ev_rank_score" in out
        else "rank_mlp_direct"
    )
    rank = _num(out, rank_col, 0.0)
    hit = _num(out, "hit_probability", 0.5)
    blocked = (
        out["side_name"].astype(str).eq("long")
        & out["archetype_policy_key"].astype(str).eq(BLACKLIST)
    ).to_numpy()
    rank[blocked] = -1.0
    return out, rank, hit


def _metrics(
    frame: pd.DataFrame,
    rank: np.ndarray,
    hit: np.ndarray,
    name: str,
) -> dict[str, float]:
    budget = max(1, int(np.ceil(0.10 * len(frame))))
    return _score_metrics(frame, rank, hit, name, budget)


def _breakdowns(
    frame: pd.DataFrame, rank: np.ndarray, name: str
) -> pd.DataFrame:
    budget = max(1, int(np.ceil(0.10 * len(frame))))
    selected = frame.loc[_top10_mask(rank, budget)].copy()
    selected["month"] = pd.to_datetime(selected["__ts__"], utc=True).dt.strftime("%Y-%m")
    day = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    selected["week_start"] = day - pd.to_timedelta(day.dt.weekday.to_numpy(), unit="D")
    reports: list[pd.DataFrame] = []
    for scope, cols in (
        ("month", ["month"]),
        ("week", ["week_start"]),
        ("side", ["side_name"]),
        ("side_archetype", ["side_name", "archetype_policy_key"]),
    ):
        report = selected.groupby(cols, observed=True).agg(
            trades=("ev_after_1pct", "size"),
            mean_net_ev=("ev_after_1pct", "mean"),
            sum_net_ev=("ev_after_1pct", "sum"),
            clean_rate=("clean_exec", "mean"),
            bad_mae_rate=("full_path_bad_mae_1r", "mean"),
            timeout_rate=("timeout", "mean"),
        ).reset_index()
        report["scope"] = scope
        report["arm"] = name
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "data_perp/reports/meta_market_state_encoder_ablation_evobjective_"
            "mlp_direct_conservative24_20260712_v10/oos_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_market_state_mlp_causal8d_ablation_20260713_v1"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, rank, hit = _prepare(pd.read_parquet(args.predictions))
    ts = pd.to_datetime(frame["__ts__"], utc=True)
    tune_mask = ts.lt(pd.Timestamp("2026-06-01", tz="UTC")).to_numpy()
    test_mask = ts.ge(pd.Timestamp("2026-06-01", tz="UTC")).to_numpy()
    tune = frame.loc[tune_mask].reset_index(drop=True)
    test = frame.loc[test_mask].reset_index(drop=True)
    tune_rank, tune_hit = rank[tune_mask], hit[tune_mask]
    test_rank, test_hit = rank[test_mask], hit[test_mask]
    tune_budget = max(1, int(np.ceil(0.10 * len(tune))))
    baseline_tune = _metrics(tune, tune_rank, tune_hit, "no_8d")
    search_rows: list[dict[str, float]] = []
    strengths = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
    for strength in strengths:
        adjusted = _causal_8d_residual_overlay(
            tune, tune_rank, tune_hit, strength, tune_budget
        )
        metric = _metrics(tune, adjusted, tune_hit, f"causal8d_{strength:.2f}")
        metric["strength"] = strength
        metric["objective"] = _objective(metric, baseline_tune)
        search_rows.append(metric)
    search = pd.DataFrame(search_rows).sort_values("objective", ascending=False)
    best_strength = float(search.iloc[0]["strength"])

    # Apply the frozen strength to the complete chronological stream so June
    # can consume only already-resolved April-May selected outcomes.
    full_budget = max(1, int(np.ceil(0.10 * len(frame))))
    full_adjusted = _causal_8d_residual_overlay(
        frame, rank, hit, best_strength, full_budget
    )
    test_adjusted = full_adjusted[test_mask]
    baseline_test = _metrics(test, test_rank, test_hit, "no_8d")
    smoother_test = _metrics(
        test, test_adjusted, test_hit, f"causal8d_{best_strength:.2f}"
    )
    summary = pd.DataFrame([baseline_test, smoother_test])
    summary["delta_mean_ev_vs_no8d"] = (
        summary["mean_ev_after_1pct"] - baseline_test["mean_ev_after_1pct"]
    )
    summary["delta_worst_week_vs_no8d"] = (
        summary["worst_week_ev"] - baseline_test["worst_week_ev"]
    )
    summary["delta_worst_month_vs_no8d"] = (
        summary["worst_month_ev"] - baseline_test["worst_month_ev"]
    )
    breakdown = pd.concat(
        [
            _breakdowns(test, test_rank, "no_8d"),
            _breakdowns(test, test_adjusted, f"causal8d_{best_strength:.2f}"),
        ],
        ignore_index=True,
    )
    search.to_csv(args.output_dir / "tuning_search_april_may.csv", index=False)
    summary.to_csv(args.output_dir / "june_summary.csv", index=False)
    breakdown.to_csv(args.output_dir / "june_breakdowns.csv", index=False)
    manifest = {
        "schema": "market_state_mlp_causal8d_ablation_v1",
        "parent_policy": POLICY_ID,
        "legacy_regime_calibrator": "disabled",
        "blacklist": [f"long||{BLACKLIST}"],
        "tuning_period": ["2026-04-01", "2026-06-01"],
        "untouched_evaluation_period": ["2026-06-01", "2026-07-01"],
        "best_strength": best_strength,
        "activity_contract": "same global top-10 row count in each comparison",
        "causality": "8-day side x archetype residual mean uses prior admitted top-10 days only",
        "source_predictions": str(args.predictions),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
