#!/usr/bin/env python3
"""Reliability and matched-delta report for the MC1 Adaptive Exit retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/adaptive_exit_mc1_population_state_retraining_20260814_v1"
ARMS = (
    "PARENT_SIMPLE_POLICY", "C0_EXISTING_V1",
    "C0_MATCHED_A5_TRAIN_MC1_SCORE", "C1_MC1_PARENT_STATES",
    "C2_MC1_V1_VISITED_STATES", "C3_MC1_MIXED_70_30",
)
SEED = 20260814


def _accepted(source: Path, arm: str) -> pd.DataFrame:
    rows = pd.read_parquet(source / f"portfolio_{arm}" / "decisions.parquet")
    rows = rows[rows.accepted.fillna(False)].copy()
    rows["timestamp"] = pd.to_datetime(rows.timestamp, utc=True)
    rows["week"] = rows.timestamp.dt.strftime("%G-W%V")
    rows["month"] = rows.timestamp.dt.strftime("%Y-%m")
    rows["net_bps"] = pd.to_numeric(rows.position_net_return, errors="coerce") * 10_000.0
    return rows


def _block_bootstrap(weekly: pd.DataFrame, challenger: str, control: str, draws: int = 10_000):
    wide_sum = weekly.pivot(index="week", columns="arm", values="net_sum_bps").fillna(0.0)
    wide_n = weekly.pivot(index="week", columns="arm", values="trades").fillna(0.0)
    weeks = wide_sum.index.intersection(wide_n.index)
    rng = np.random.default_rng(SEED)
    delta = np.empty(draws, dtype=float)
    for draw in range(draws):
        take = rng.integers(0, len(weeks), len(weeks))
        selected = weeks[take]
        challenger_ev = wide_sum.loc[selected, challenger].sum() / max(
            wide_n.loc[selected, challenger].sum(), 1
        )
        control_ev = wide_sum.loc[selected, control].sum() / max(
            wide_n.loc[selected, control].sum(), 1
        )
        delta[draw] = challenger_ev - control_ev
    return {
        "challenger": challenger, "control": control,
        "weekly_blocks": len(weeks), "draws": draws,
        "delta_bps": float(
            wide_sum[challenger].sum() / wide_n[challenger].sum()
            - wide_sum[control].sum() / wide_n[control].sum()
        ),
        "ci025_bps": float(np.quantile(delta, .025)),
        "ci50_bps": float(np.quantile(delta, .50)),
        "ci975_bps": float(np.quantile(delta, .975)),
        "probability_delta_positive": float(np.mean(delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    frames = {arm: _accepted(args.source, arm) for arm in ARMS}
    weekly_parts = []
    monthly_parts = []
    for arm, rows in frames.items():
        weekly = rows.groupby("week").net_bps.agg(trades="size", net_ev_bps="mean", net_sum_bps="sum").reset_index()
        weekly["arm"] = arm
        weekly_parts.append(weekly)
        monthly = rows.groupby("month").net_bps.agg(trades="size", net_ev_bps="mean", net_sum_bps="sum").reset_index()
        monthly["arm"] = arm
        monthly_parts.append(monthly)
    weekly = pd.concat(weekly_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    weekly.to_parquet(args.source / "portfolio_weekly_metrics.parquet", index=False)
    monthly.to_parquet(args.source / "portfolio_monthly_metrics.parquet", index=False)

    control = weekly[weekly.arm.eq("C0_MATCHED_A5_TRAIN_MC1_SCORE")].set_index("week")
    stability = []
    for arm in ARMS:
        local = weekly[weekly.arm.eq(arm)].set_index("week")
        common = local.index.intersection(control.index)
        delta = local.loc[common, "net_ev_bps"] - control.loc[common, "net_ev_bps"]
        stability.append({
            "arm": arm, "weeks": len(local),
            "positive_weeks": int(local.net_ev_bps.gt(0).sum()),
            "negative_weeks": int(local.net_ev_bps.lt(0).sum()),
            "worst_week_ev_bps": float(local.net_ev_bps.min()),
            "median_week_ev_bps": float(local.net_ev_bps.median()),
            "weeks_vs_matched_c0": len(common),
            "positive_uplift_weeks_vs_matched_c0": int(delta.gt(0).sum()),
            "negative_uplift_weeks_vs_matched_c0": int(delta.lt(0).sum()),
            "median_week_uplift_vs_matched_c0_bps": float(delta.median()),
            "worst_week_uplift_vs_matched_c0_bps": float(delta.min()),
        })
    stability = pd.DataFrame(stability)
    stability.to_parquet(args.source / "portfolio_weekly_stability.parquet", index=False)

    bootstrap = pd.DataFrame([
        _block_bootstrap(weekly, arm, "C0_MATCHED_A5_TRAIN_MC1_SCORE")
        for arm in ("C1_MC1_PARENT_STATES", "C2_MC1_V1_VISITED_STATES", "C3_MC1_MIXED_70_30")
    ])
    bootstrap.to_parquet(args.source / "weekly_block_bootstrap.parquet", index=False)

    replay = pd.read_parquet(args.source / "oof_replay.parquet")
    incumbent = replay[
        replay.arm.eq("C0_MATCHED_A5_TRAIN_MC1_SCORE")
    ].drop_duplicates("candidate_id")
    incumbent = incumbent[["candidate_id", "adaptive_net_bps"]].rename(
        columns={"adaptive_net_bps": "C0_MATCHED_A5_TRAIN_MC1_SCORE"}
    )
    paired = incumbent
    for arm in ("C1_MC1_PARENT_STATES", "C2_MC1_V1_VISITED_STATES", "C3_MC1_MIXED_70_30"):
        local = replay[replay.arm.eq(arm)].drop_duplicates("candidate_id")
        paired = paired.merge(
            local[["candidate_id", "adaptive_net_bps"]].rename(columns={"adaptive_net_bps": arm}),
            on="candidate_id", how="outer",
        )
    paired.to_parquet(args.source / "paired_candidate_outcomes.parquet", index=False)
    paired_summary = []
    for arm in ("C1_MC1_PARENT_STATES", "C2_MC1_V1_VISITED_STATES", "C3_MC1_MIXED_70_30"):
        common = paired.dropna(subset=["C0_MATCHED_A5_TRAIN_MC1_SCORE", arm])
        delta = common[arm] - common.C0_MATCHED_A5_TRAIN_MC1_SCORE
        paired_summary.append({
            "arm": arm, "common_candidates": len(common),
            "mean_delta_vs_matched_c0_bps": float(delta.mean()),
            "median_delta_vs_matched_c0_bps": float(delta.median()),
            "positive_delta_fraction": float(delta.gt(0).mean()),
        })
    paired_summary = pd.DataFrame(paired_summary)
    paired_summary.to_parquet(args.source / "paired_candidate_summary.parquet", index=False)

    parity = pd.read_parquet(args.source / "stored_outcome_parity.parquet")
    valid = parity.difference_bps.abs().le(.01)
    parity_valid = pd.DataFrame([{
        "population_rows": len(parity), "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "valid_mae_bps": float(parity.loc[valid, "difference_bps"].abs().mean()),
        "valid_max_ae_bps": float(parity.loc[valid, "difference_bps"].abs().max()),
    }])
    parity_valid.to_parquet(args.source / "valid_parent_policy_parity_summary.parquet", index=False)
    report = {
        "status": "complete", "promotion": "none",
        "comparison": "C1/C2/C3 versus frozen C0 on common Jul-2025 through Jul-2026 portfolio window",
        "weekly_stability": stability.to_dict("records"),
        "weekly_block_bootstrap": bootstrap.to_dict("records"),
        "paired_candidate_summary": paired_summary.to_dict("records"),
        "parity": parity_valid.to_dict("records")[0],
    }
    (args.source / "reliability_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
