#!/usr/bin/env python3
"""Causal hard-block ablation for local failure-detector risk scores.

Scores are ranked only against strictly prior scores in the same
representation x side x archetype cell.  This is deliberately a hard block
test: it reports the PnL cost/benefit of removing high-risk admitted rows and
does not change rank, size, or exits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_POLICY = Path("data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/joint_layered_wallet80_holdeff_l2_20260718_v5/july_01_10_selected_ledger.parquet")
DEFAULT_SCORES = Path("data_perp/reports/prospective_failure_mode_detection_20260719_v7_three_year/local_oos_predictions.parquet")
DEFAULT_OUTPUT = Path("data_perp/reports/failure_detector_hard_block_ablation_20260719_v1")


def _unprefix(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"^(long|short)__", "", regex=True)


def _causal_rank(frame: pd.DataFrame) -> pd.Series:
    """Prior-only percentile rank, with no same-day score information."""

    result = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    for _, local in frame.groupby(
        ["representation", "side_name", "archetype_policy_key"], observed=True, sort=False
    ):
        local = local.sort_values("day", kind="stable")
        prior: list[float] = []
        for day, same_day in local.groupby("day", observed=True, sort=True):
            del day
            if len(prior) >= 20:
                values = np.asarray(prior, dtype=np.float64)
                current = same_day["risk"].to_numpy(np.float64)
                result.loc[same_day.index] = (
                    np.searchsorted(np.sort(values), current, side="right") / len(values)
                ).astype(np.float32)
            prior.extend(same_day["risk"].dropna().astype(float).tolist())
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    policy = pd.read_parquet(args.policy)
    policy["timestamp"] = pd.to_datetime(policy["timestamp"], utc=True)
    policy["day"] = policy["timestamp"].dt.floor("D")
    policy["archetype_policy_key"] = _unprefix(policy["policy_archetype"])
    scores = pd.read_parquet(args.scores)
    scores["day"] = pd.to_datetime(scores["day"], utc=True).dt.floor("D")
    # Chronological detector outputs carry explicit target metadata; native
    # representation-risk streams are already same-day negative-EV risk.
    if "failure_mode" in scores:
        mask = scores["failure_mode"].eq("negative_ev_day")
        if "target_horizon_days" in scores:
            mask &= scores["target_horizon_days"].eq(0)
        scores = scores.loc[
            mask, ["day", "side_name", "archetype_policy_key", "risk"]
        ].copy()
    else:
        scores = scores.loc[
            :, ["day", "side_name", "archetype_policy_key", "risk"]
            + (["representation"] if "representation" in scores else [])
        ].copy()
    if "representation" not in scores:
        scores["representation"] = str(args.representation)
    scores["risk_rank_prior"] = _causal_rank(scores)
    joined = policy.merge(
        scores,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    joined["score_available"] = joined["risk_rank_prior"].notna()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    rows = []
    detail = []
    for threshold in thresholds:
        blocked = joined["score_available"] & joined["risk_rank_prior"].ge(threshold)
        retained = joined.loc[~blocked]
        base = pd.to_numeric(joined["net_return"], errors="coerce")
        kept = pd.to_numeric(retained["net_return"], errors="coerce")
        rows.append({
            "representation": str(args.representation), "risk_rank_threshold": threshold,
            "rows": int(len(joined)), "score_coverage": float(joined["score_available"].mean()),
            "blocked_rows": int(blocked.sum()), "blocked_share": float(blocked.mean()),
            "retained_rows": int(len(retained)), "baseline_ev_per_trade": float(base.mean()),
            "retained_ev_per_trade": float(kept.mean()),
            "delta_ev_per_trade": float(kept.mean() - base.mean()),
            "baseline_total_pnl": float(pd.to_numeric(joined["pnl"], errors="coerce").sum()),
            "retained_total_pnl": float(pd.to_numeric(retained["pnl"], errors="coerce").sum()),
            "baseline_worst_day": float(joined.groupby("day", observed=True)["net_return"].mean().min()),
            "retained_worst_day": float(retained.groupby("day", observed=True)["net_return"].mean().min()),
        })
        cell = joined.assign(blocked=blocked).groupby(
            ["side_name", "archetype_policy_key"], observed=True
        ).agg(
            rows=("net_return", "size"), blocked_rows=("blocked", "sum"),
            baseline_ev=("net_return", "mean"),
        ).reset_index()
        retained_cell = retained.groupby(["side_name", "archetype_policy_key"], observed=True)["net_return"].mean()
        cell["retained_ev"] = [retained_cell.get((r.side_name, r.archetype_policy_key), np.nan) for r in cell.itertuples()]
        cell["delta_ev"] = cell["retained_ev"] - cell["baseline_ev"]
        cell["representation"] = str(args.representation)
        cell["risk_rank_threshold"] = threshold
        detail.append(cell)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.concat(detail, ignore_index=True).to_csv(output / "side_archetype.csv", index=False)
    joined.to_parquet(output / "joined_policy_scores.parquet", index=False)
    manifest = {
        "schema": "failure_detector_hard_block_ablation_v1",
        "representation": str(args.representation), "thresholds": thresholds,
        "policy": str(Path(args.policy).resolve()), "scores": str(Path(args.scores).resolve()),
        "causal_rank_contract": "Each day is ranked only versus strictly earlier same representation x side x archetype OOS scores; insufficient prior support remains unblocked.",
        "scope_warning": "This run is a consensus-detector control unless scores come from one representation-specific detector stream.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--representation", default="frozen_consensus_detector_control")
    parser.add_argument("--thresholds", default="0.95,0.96,0.97,0.98,0.99")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
