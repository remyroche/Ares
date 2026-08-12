#!/usr/bin/env python3
"""Consolidate the long-only C3 and self-distillation research funnel.

The report compares separately selected global tails, but uncertainty is
estimated with a paired calendar-day bootstrap.  It also materialises base
calibration, K9-regime economics, score-decile economics, selection overlap,
and base feature-importance drift for the exact matched D0/D2 contracts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
BASE_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_self_distillation_base_d2_top20_boost15_"
    "long_2024_jul2026_20260810_v1"
)
STACKS = {
    (2025, "D0"): ROOT / (
        "data_perp/artifacts/strict_r3_self_distillation_matched_d0_fullstack_"
        "long_2025_janjul_20260810_v1"
    ),
    (2025, "D2_top20_boost1.5"): ROOT / (
        "data_perp/artifacts/strict_r3_self_distillation_combined_base_d2_residual_d0_"
        "fullstack_long_2025_janjul_20260810_v1"
    ),
    (2026, "D0"): ROOT / (
        "data_perp/artifacts/strict_r3_self_distillation_matched_d0_fullstack_"
        "long_2026_janjul_exact_policy_20260810_v1"
    ),
    (2026, "D2_top20_boost1.5"): ROOT / (
        "data_perp/artifacts/strict_r3_self_distillation_combined_base_d2_residual_d0_"
        "fullstack_long_2026_janjul_exact_policy_20260810_v1"
    ),
}


def _load_stack(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path / "predictions.parquet")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame = frame.loc[
        np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce"))
    ].copy()
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError(f"invalid stack predictions: {path}")
    return frame


def _selected(frame: pd.DataFrame, tail: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(tail * len(frame))))
    return frame.nlargest(count, "final_score", keep="first").copy()


def _valid_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()


def _paired_day_bootstrap(
    control: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    draws: int = 5_000,
    seed: int = 1729,
) -> dict[str, float]:
    def daily(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        local = frame.assign(day=frame["__decision_ts__"].dt.floor("D"))
        return local.groupby("day", sort=True)["policy_net_bps"].agg(
            **{f"{prefix}_sum": "sum", f"{prefix}_count": "size"},
        )

    joined = daily(control, "control").join(daily(challenger, "challenger"), how="outer").fillna(0.0)
    values = joined.to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].sum(axis=1)
    control_ev = sampled[:, 0] / np.maximum(sampled[:, 1], 1.0)
    challenger_ev = sampled[:, 2] / np.maximum(sampled[:, 3], 1.0)
    delta = challenger_ev - control_ev
    return {
        "delta_mean_bps": float(delta.mean()),
        "delta_ci025_bps": float(np.quantile(delta, 0.025)),
        "delta_ci975_bps": float(np.quantile(delta, 0.975)),
        "probability_delta_positive": float((delta > 0.0).mean()),
        "bootstrap_days": int(len(values)),
        "bootstrap_draws": int(draws),
    }


def _stack_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    loaded = {(year, arm): _load_stack(path) for (year, arm), path in STACKS.items()}
    for (year, arm), frame in loaded.items():
        membership = [column for column in frame if column.endswith("__membership")]
        frame["dominant_k9"] = frame[membership].to_numpy(float).argmax(axis=1)
        ordered = frame.sort_values(["final_score", "candidate_id"], kind="stable").copy()
        ordered["score_decile"] = np.minimum(np.arange(len(ordered)) * 10 // len(ordered), 9)
        for decile, block in ordered.groupby("score_decile", sort=True):
            valid = _valid_outcomes(block)
            decile_rows.append({
                "year": year, "arm": arm, "score_decile": int(decile),
                "selected_score_rows": len(block), "valid_outcomes": len(valid),
                "outcome_coverage": len(valid) / max(len(block), 1),
                "net_bps_per_trade": float(valid["policy_net_bps"].mean()),
                "positive_rate": float(valid["policy_net_bps"].gt(0).mean()),
            })
        for tail in TAILS:
            selected_score = _selected(frame, tail)
            selected = _valid_outcomes(selected_score)
            metric_rows.append({
                "year": year, "arm": arm, "tail": tail,
                "population_rows": len(frame), "selected_score_rows": len(selected_score),
                "valid_outcomes": len(selected),
                "outcome_coverage": len(selected) / max(len(selected_score), 1),
                "trades": len(selected),
                "net_bps_per_trade": float(selected["policy_net_bps"].mean()),
                "positive_rate": float(selected["policy_net_bps"].gt(0).mean()),
            })
            for regime, block in selected.groupby("dominant_k9", sort=True):
                regime_rows.append({
                    "year": year, "arm": arm, "tail": tail, "dominant_k9": int(regime),
                    "trades": len(block), "net_bps_per_trade": float(block["policy_net_bps"].mean()),
                    "positive_rate": float(block["policy_net_bps"].gt(0).mean()),
                })
    for year in (2025, 2026):
        control = loaded[(year, "D0")]
        challenger = loaded[(year, "D2_top20_boost1.5")]
        for tail in TAILS:
            selected_control_all = _selected(control, tail)
            selected_challenger_all = _selected(challenger, tail)
            selected_control = _valid_outcomes(selected_control_all)
            selected_challenger = _valid_outcomes(selected_challenger_all)
            control_ids = set(selected_control_all["candidate_id"])
            challenger_ids = set(selected_challenger_all["candidate_id"])
            overlap_rows.append({
                "year": year, "tail": tail, "control_trades": len(control_ids),
                "challenger_trades": len(challenger_ids), "intersection": len(control_ids & challenger_ids),
                "jaccard": len(control_ids & challenger_ids) / max(len(control_ids | challenger_ids), 1),
            })
            bootstrap_rows.append({
                "year": year, "tail": tail,
                **_paired_day_bootstrap(selected_control, selected_challenger, seed=1729 + year),
            })
    return (
        pd.DataFrame(metric_rows), pd.DataFrame(bootstrap_rows), pd.DataFrame(overlap_rows),
        pd.DataFrame(regime_rows), pd.DataFrame(decile_rows),
    )


def _base_calibration() -> pd.DataFrame:
    frame = pd.read_parquet(BASE_LEDGER / "base_oof_predictions.parquet")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    rows: list[dict[str, object]] = []
    for (year, arm), local in frame.groupby([frame["__decision_ts__"].dt.year, "arm"], sort=True):
        if year not in (2025, 2026) or arm not in ("D0", "D2_top20_boost1.5"):
            continue
        local = local.loc[local["r3_class"].notna()].copy()
        local["calibration_bin"] = pd.qcut(
            local["p_clear"].rank(method="first"), 10, labels=False, duplicates="drop",
        )
        for bucket, block in local.groupby("calibration_bin", sort=True):
            rows.append({
                "year": int(year), "arm": str(arm), "calibration_bin": int(bucket),
                "rows": len(block), "mean_p_clear": float(block["p_clear"].mean()),
                "observed_clear_rate": float(block["r3_class"].eq(2).mean()),
            })
    return pd.DataFrame(rows)


def _feature_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(BASE_LEDGER / "feature_importance_by_fold.parquet")
    frame["year"] = frame["held_month"].str[:4].astype(int)
    frame = frame.loc[
        frame["year"].isin([2025, 2026])
        & frame["arm"].isin(["D0", "D2_top20_boost1.5"])
    ].copy()
    frame["gain_share"] = frame["gain_importance"] / frame.groupby(
        ["arm", "held_month"], sort=False
    )["gain_importance"].transform("sum").clip(lower=1e-12)
    top = frame.sort_values(
        ["arm", "held_month", "gain_share"], ascending=[True, True, False], kind="stable",
    ).groupby(["arm", "held_month"], sort=False).head(20)
    summary: list[dict[str, object]] = []
    for arm, local in top.groupby("arm", sort=True):
        months = sorted(local["held_month"].unique())
        for left, right in zip(months, months[1:]):
            a = set(local.loc[local["held_month"].eq(left), "feature"])
            b = set(local.loc[local["held_month"].eq(right), "feature"])
            summary.append({
                "arm": arm, "left_month": left, "right_month": right,
                "top20_jaccard": len(a & b) / max(len(a | b), 1),
            })
    return top, pd.DataFrame(summary)


def _fmt_tail(value: float) -> str:
    return f"{100 * value:g}%"


def _write_report(
    out: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    overlap: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    lines = [
        "# Strict-R3 Long-Only C3 and Self-Distillation Funnel — 2026-08-10",
        "",
        "## Decision",
        "",
        "Promote the six-month downstream window, four-week refit cadence, three-month C3 "
        "geometry burn-in, and the base D2 Top-20% robust-clear curriculum at 1.5x. The "
        "two-week cadence won the 2025 development screen but failed the 2026 worst-month "
        "transport gate; four weeks is therefore retained. Keep ordinary residual weighting: "
        "residual D3 did not survive the complete stack.",
        "",
        "All reported outcomes use the pre-2025 SimplePolicyOptimiser winner: stop 4.1520 ATR, "
        "trailing activation 2.3262 ATR, giveback 0.10237 ATR, H12 timeout, and 100 bps cost "
        "exactly once. The executable replay uses causal side-local 21-day mapped EV >= +50 "
        "bps, followed by portfolio constraints.",
        "",
        "## Matched full-stack economics",
        "",
        "| Year | Arm | Tail | Selected rows | Valid outcomes | Coverage | Net bps/trade | Positive rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.arm} | {_fmt_tail(row.tail)} | "
            f"{row.selected_score_rows:,} | {row.valid_outcomes:,} | "
            f"{100 * row.outcome_coverage:.1f}% | {row.net_bps_per_trade:+.2f} | "
            f"{100 * row.positive_rate:.1f}% |"
        )
    lines.extend([
        "",
        "## Paired calendar-day bootstrap: D2 minus matched D0",
        "",
        "| Year | Tail | Mean delta | 95% CI | P(delta > 0) | Top-set Jaccard |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    joined = bootstrap.merge(overlap[["year", "tail", "jaccard"]], on=["year", "tail"])
    for row in joined.itertuples(index=False):
        lines.append(
            f"| {row.year} | {_fmt_tail(row.tail)} | {row.delta_mean_bps:+.2f} bps | "
            f"[{row.delta_ci025_bps:+.2f}, {row.delta_ci975_bps:+.2f}] | "
            f"{100 * row.probability_delta_positive:.1f}% | {row.jaccard:.3f} |"
        )
    lines.extend([
        "",
        "## Reliability notes",
        "",
        "- D0 and D2 are matched on candidate identities, base/residual parameters, the 240k "
        "confirmation cap, C3 geometry, policy outcomes, causal admission, and portfolio rules.",
        "- Every global tail is selected from all finite-score candidates before outcome "
        "coverage is inspected. Missing or invalid future paths never alter the selected-set "
        "denominator; EV and hit rate are reported only on the selected rows with valid outcomes.",
        "- D2 changes only the base training weight: robust-clear rows in the teacher's global "
        "Top-20% receive a 1.5x pre-normalisation boost; weights are projected to mean one and "
        "capped to [0.25, 4].",
        "- The teacher is the prior strict-prequential base rank42. No held-month or per-timestamp "
        "ranking is used.",
        "- The paired bootstrap is a dependence-aware diagnostic, not untouched evidence: 2025 "
        "selected the curriculum; 2026 is the later confirmation period.",
        "- The top-20 base feature drift audit has mean consecutive-month Jaccard of "
        f"{drift.groupby('arm')['top20_jaccard'].mean().to_dict()}.",
        "",
        "## Artifact map",
        "",
        "The companion parquet files contain exact metrics, bootstrap draws summary, selected-set "
        "overlap, K9-regime economics, score-decile economics, base calibration, and feature drift.",
    ])
    (out / "STRICT_R3_C3_SELF_DISTILLATION_LONG_ONLY_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    metrics, bootstrap, overlap, regime, decile = _stack_audit()
    calibration = _base_calibration()
    importance, drift = _feature_drift()
    artifacts = {
        "matched_full_stack_metrics.parquet": metrics,
        "paired_day_bootstrap.parquet": bootstrap,
        "selected_set_overlap.parquet": overlap,
        "k9_regime_tail_economics.parquet": regime,
        "score_decile_economics.parquet": decile,
        "base_calibration_curves.parquet": calibration,
        "base_top20_feature_importance.parquet": importance,
        "base_feature_importance_drift.parquet": drift,
    }
    for name, frame in artifacts.items():
        frame.to_parquet(args.out_dir / name, index=False, compression="zstd")
    _write_report(args.out_dir, metrics, bootstrap, overlap, drift)
    manifest = {
        "schema": "strict_r3_c3_self_distillation_long_only_report_v2",
        "side": "long",
        "stacks": {f"{year}_{arm}": str(path) for (year, arm), path in STACKS.items()},
        "base_ledger": str(BASE_LEDGER),
        "bootstrap_unit": "calendar_day",
        "bootstrap_draws": 5_000,
        "global_tail_selection": True,
        "global_tail_selection_precedes_outcome_coverage": True,
        "status": "complete",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
