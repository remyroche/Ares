#!/usr/bin/env python3
"""Evaluate a frozen short-default challenger with day/event-level evidence.

This script is validation-only. It consumes frozen V11 and challenger ranks,
does not search thresholds, and writes a reproducible credibility packet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.challenger_credibility import (
    PosteriorConfig,
    bayesian_bootstrap_contract_probability,
    daily_decision_deltas,
    hierarchical_student_t_posterior,
    leave_group_out,
)


GROUP = ("short", "short_default_clean_path")
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load(v11_dir: Path, challenger_dir: Path) -> pd.DataFrame:
    parent = pd.read_parquet(v11_dir / "oos_predictions.parquet")
    challenger = pd.read_parquet(challenger_dir / "oos_replication_predictions.parquet")
    parent["__ts__"] = pd.to_datetime(parent["__ts__"], utc=True)
    challenger["__ts__"] = pd.to_datetime(challenger["__ts__"], utc=True)
    parent = parent.loc[
        parent["side_name"].eq(GROUP[0])
        & parent["archetype_policy_key"].eq(GROUP[1]),
        [*KEYS, "parent_rank_v9_residual_error_overlay", "ev_after_1pct", "clean_exec", "adverse_calendar_cell"],
    ]
    challenger = challenger.loc[
        challenger["side_name"].eq(GROUP[0])
        & challenger["archetype_policy_key"].eq(GROUP[1]),
        [*KEYS, "frozen_short_default_uncertainty_rank"],
    ]
    return parent.merge(challenger, on=KEYS, how="inner", validate="one_to_one")


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _load(args.v11_dir, args.challenger_dir)
    daily = daily_decision_deltas(
        rows,
        parent_rank="parent_rank_v9_residual_error_overlay",
        challenger_rank="frozen_short_default_uncertainty_rank",
    )
    daily.to_csv(args.output / "daily_decision_deltas.csv", index=False)
    posterior = hierarchical_student_t_posterior(
        daily,
        config=PosteriorConfig(draws=args.draws, burn_in=args.burn_in, seed=args.seed),
    )
    posterior.to_parquet(args.output / "student_t_hierarchical_posterior.parquet", index=False, compression="zstd")
    minimum = float(args.minimum_ev_pp) / 10_000.0
    contract = bayesian_bootstrap_contract_probability(
        daily, draws=args.bootstrap_draws, seed=args.seed + 1, minimum_ev_per_trade=minimum
    )
    leave_out = pd.concat(
        [
            leave_group_out(daily.assign(day_key=daily["day"].astype(str)), "day_key"),
            leave_group_out(daily.loc[daily["event_block"].ne("normal")], "event_block"),
            leave_group_out(daily, "month"),
        ],
        ignore_index=True,
        copy=False,
    )
    leave_out.to_csv(args.output / "leave_one_day_event_month_out.csv", index=False)
    summary = {
        "schema": "short_default_challenger_credibility_v1",
        "scope": {"side_name": GROUP[0], "archetype_policy_key": GROUP[1], "rows": int(len(rows)), "days": int(len(daily))},
        "posterior": {
            "p_mu_gt_zero": float((posterior["mu"] > 0.0).mean()),
            "p_mu_gt_minimum": float((posterior["mu"] > minimum).mean()),
            "posterior_mu_mean": float(posterior["mu"].mean()),
            "posterior_mu_q05": float(posterior["mu"].quantile(0.05)),
            "posterior_mu_q95": float(posterior["mu"].quantile(0.95)),
            "sampler_acceptance_rate": float(posterior.attrs["acceptance_rate"]),
            "model": "Student-t daily deltas with partial pooling by month and adverse/normal event family",
        },
        "joint_contract": contract,
        "aggregate": {
            "delta_total_ev": float(daily["delta_total_ev"].sum()),
            "delta_ev_per_trade": float(daily["challenger_total_ev"].sum() / max(daily["challenger_selected"].sum(), 1) - daily["parent_total_ev"].sum() / max(daily["parent_selected"].sum(), 1)),
            "delta_clean_precision": float(daily["challenger_clean_sum"].sum() / max(daily["challenger_selected"].sum(), 1) - daily["parent_clean_sum"].sum() / max(daily["parent_selected"].sum(), 1)),
            "activity_ratio": float(daily["challenger_selected"].sum() / max(daily["parent_selected"].sum(), 1)),
            "largest_positive_day_share": float(
                daily["delta_total_ev"].clip(lower=0.0).max()
                / max(daily["delta_total_ev"].clip(lower=0.0).sum(), 1e-12)
            ),
        },
        "leave_out_flags": {
            "total_ev_sign_reversals": int(leave_out["total_ev_sign_reversal"].sum()),
            "ev_per_trade_sign_reversals": int(leave_out["ev_per_trade_sign_reversal"].sum()),
            "max_abs_event_block_influence": float(
                leave_out.loc[leave_out["group_column"].eq("event_block"), "influence_share"].abs().max()
            ) if leave_out["group_column"].eq("event_block").any() else None,
        },
        "leakage_contract": (
            "Uses only frozen V11/challenger OOS predictions. Days are the evidence unit; "
            "no threshold, model, or feature is refit. Event blocks are derived solely from "
            "realized calendar labels for retrospective validation and are never inference inputs."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(_safe(summary), indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-dir", type=Path, required=True)
    parser.add_argument("--challenger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=8_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--minimum-ev-pp", type=float, default=2.0, help="Meaningful EV/trade hurdle in percentage points.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
