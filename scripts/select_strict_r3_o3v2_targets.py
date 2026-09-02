#!/usr/bin/env python3
"""Predeclared development selection and forward audit for O3-v2 targets.

The target funnel is a broad, strict-OOF diagnostic.  This small second pass
prevents it becoming a hindsight target lottery: it selects at most two target
*concepts* from a declared development block, then emits a later forward block
without using it to alter the selection.  The candidate arm set is itself a
declared contract: a deliberately bounded target screen must not be rejected
merely because it did not run legacy, out-of-scope arms.
them to alter the selection.  It never fits a model, writes a score, or
touches MC1/live artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_o3v2_target_selection_v1"
TAIL_WEIGHTS = {0.01: .40, 0.02: .35, 0.05: .25}
CONCEPT = {
    "T1_economic_residual_lambdarank": "economic_residual",
    "T2_economic_residual_ordinal": "economic_residual",
    "T3_pair_residual_lambdarank": "economic_residual",
    "T4_hard_inversion_lambdarank": "hard_inversion",
    "T5_rank_error_lambdarank": "rank_error",
    "T6_rank_error_ordinal": "rank_error",
    "T7_exit4_lambdarank": "policy_state",
    "T8_exit5_lambdarank": "policy_state",
    "T9_exit5_ordinal": "policy_state",
}
CONTROL = "T0_current_o3_control"
PRIMARY_SCORE = "o3v2_rank_75_25"
DEFAULT_CANDIDATE_ARMS = (
    "T1_economic_residual_lambdarank",
    "T2_economic_residual_ordinal",
    "T4_hard_inversion_lambdarank",
    "T6_rank_error_ordinal",
    "T8_exit5_lambdarank",
    "T9_exit5_ordinal",
)


def _months(raw: str) -> tuple[str, ...]:
    result = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not result:
        raise ValueError("at least one YYYY-MM month is required")
    return result


def _arms(raw: str) -> tuple[str, ...]:
    result = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not result:
        raise ValueError("at least one candidate arm is required")
    unknown = set(result) - set(CONCEPT)
    if unknown:
        raise ValueError(f"unknown candidate arms: {sorted(unknown)}")
    if len(set(result)) != len(result):
        raise ValueError("candidate arms must be unique")
    return result


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _delta(metrics: pd.DataFrame, months: tuple[str, ...]) -> pd.DataFrame:
    if "score" not in metrics.columns:
        raise AssertionError("target metrics lack the declared score identity")
    primary = metrics.loc[metrics["score"].eq(PRIMARY_SCORE)].copy()
    if primary.empty:
        raise AssertionError(f"target metrics lack declared primary score {PRIMARY_SCORE}")
    base = primary.loc[
        primary["arm"].eq(CONTROL) & primary["month"].isin(months) & primary["tail"].isin(TAIL_WEIGHTS),
        ["month", "tail", "net_ev_bps_per_trade", "policy_rank_ic"],
    ].rename(columns={"net_ev_bps_per_trade": "control_net_ev_bps_per_trade", "policy_rank_ic": "control_rank_ic"})
    arms = primary.loc[
        primary["arm"].ne(CONTROL) & primary["month"].isin(months) & primary["tail"].isin(TAIL_WEIGHTS),
    ].copy()
    result = arms.merge(base, on=["month", "tail"], how="inner", validate="many_to_one")
    result["delta_net_ev_bps_per_trade"] = result["net_ev_bps_per_trade"] - result["control_net_ev_bps_per_trade"]
    result["concept"] = result["arm"].map(CONCEPT)
    if result["concept"].isna().any():
        raise AssertionError(f"unclassified target arms: {sorted(result.loc[result['concept'].isna(), 'arm'].unique())}")
    return result


def _development_table(delta: pd.DataFrame, months: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for arm, local in delta.groupby("arm", sort=True):
        weighted_by_month = []
        for month, month_part in local.groupby("month", sort=True):
            actual = month_part.set_index("tail")["delta_net_ev_bps_per_trade"]
            if set(TAIL_WEIGHTS) - set(actual.index):
                continue
            weighted_by_month.append(float(sum(TAIL_WEIGHTS[tail] * actual.loc[tail] for tail in TAIL_WEIGHTS)))
        top2 = local.loc[np.isclose(local["tail"], .02), "delta_net_ev_bps_per_trade"].to_numpy(float)
        top5 = local.loc[np.isclose(local["tail"], .05), "delta_net_ev_bps_per_trade"].to_numpy(float)
        rank_delta = (
            local.groupby("month", sort=True)["policy_rank_ic"].mean()
            - local.groupby("month", sort=True)["control_rank_ic"].mean()
        ).to_numpy(float)
        if not weighted_by_month:
            continue
        weighted = np.asarray(weighted_by_month, dtype=float)
        score = float(np.mean(weighted) - .25 * np.std(weighted) - max(0.0, -float(np.min(weighted))))
        rows.append({
            "arm": arm, "concept": str(local["concept"].iloc[0]), "development_months": ",".join(months),
            "weighted_delta_mean_bps": float(np.mean(weighted)), "weighted_delta_std_bps": float(np.std(weighted)),
            "weighted_delta_worst_month_bps": float(np.min(weighted)), "selection_score_bps": score,
            "top2_positive_months": int(np.sum(top2 > 0.0)), "top5_positive_months": int(np.sum(top5 > 0.0)),
            "top2_delta_mean_bps": float(np.mean(top2)), "top5_delta_mean_bps": float(np.mean(top5)),
            "mean_rank_ic_delta": float(np.mean(rank_delta)),
            "eligible": bool(score > 0.0 and np.sum(top2 > 0.0) >= max(2, len(months) - 1) and np.sum(top5 > 0.0) >= 2),
        })
    return pd.DataFrame(rows).sort_values("selection_score_bps", ascending=False, kind="stable").reset_index(drop=True)


def run(
    *, metrics_path: Path, out: Path, development_months: tuple[str, ...],
    forward_months: tuple[str, ...], candidate_arms: tuple[str, ...],
) -> None:
    if out.exists():
        raise FileExistsError(out)
    metrics = pd.read_parquet(metrics_path)
    expected = {CONTROL, *candidate_arms}
    missing = expected - set(metrics["arm"].unique())
    if missing:
        raise AssertionError(f"target metrics lack expected arms: {sorted(missing)}")
    development_delta = _delta(metrics, development_months)
    table = _development_table(development_delta, development_months)
    # One arm per concept prevents variants of a single label formulation from
    # occupying both scarce downstream contracts.  Tie breaks are deterministic.
    finalists = (
        table.loc[table["eligible"]]
        .sort_values(["concept", "selection_score_bps", "arm"], ascending=[True, False, True], kind="stable")
        .groupby("concept", sort=False, as_index=False).first()
        .sort_values(["selection_score_bps", "arm"], ascending=[False, True], kind="stable")
        .head(2)
        .reset_index(drop=True)
    )
    selected = finalists["arm"].tolist()
    forward_delta = _delta(metrics, forward_months)
    forward_rows = forward_delta.loc[forward_delta["arm"].isin(selected)].copy()
    summary = []
    for arm, local in forward_rows.groupby("arm", sort=True):
        for tail, part in local.groupby("tail", sort=True):
            summary.append({
                "arm": arm, "concept": str(part["concept"].iloc[0]), "tail": float(tail),
                "months": ",".join(forward_months), "delta_mean_bps": float(part["delta_net_ev_bps_per_trade"].mean()),
                "delta_worst_month_bps": float(part["delta_net_ev_bps_per_trade"].min()),
                "positive_months": int(np.sum(part["delta_net_ev_bps_per_trade"] > 0.0)),
                "control_ev_mean_bps": float(part["control_net_ev_bps_per_trade"].mean()),
                "challenger_ev_mean_bps": float(part["net_ev_bps_per_trade"].mean()),
                "rank_ic_delta_mean": float((part["policy_rank_ic"] - part["control_rank_ic"]).mean()),
            })
    out.mkdir(parents=True)
    table.to_parquet(out / "target_development_selection.parquet", index=False, compression="zstd")
    development_delta.to_parquet(out / "target_development_monthly_delta.parquet", index=False, compression="zstd")
    forward_rows.to_parquet(out / "target_forward_monthly_delta.parquet", index=False, compression="zstd")
    pd.DataFrame(summary).to_parquet(out / "target_forward_summary.parquet", index=False, compression="zstd")
    _exclusive_json(out / "selected_target_contracts.json", {
        "schema": SCHEMA, "control": CONTROL, "primary_score": PRIMARY_SCORE, "development_months": list(development_months),
        "forward_months": list(forward_months), "selection": "predeclared full-reserve development block, max one arm per target concept, max two total",
        "candidate_arms": list(candidate_arms),
        "selected": selected, "selected_concepts": finalists["concept"].tolist(),
        "held_out_rule": "forward block is diagnostic only and was never read by target selection",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-months", default="2025-11,2025-12,2026-01")
    parser.add_argument("--forward-months", default="2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--candidate-arms", default=",".join(DEFAULT_CANDIDATE_ARMS),
                        help="comma-separated predeclared target arms present in the bounded source screen")
    args = parser.parse_args()
    run(
        metrics_path=args.metrics, out=args.out,
        development_months=_months(args.development_months), forward_months=_months(args.forward_months),
        candidate_arms=_arms(args.candidate_arms),
    )


if __name__ == "__main__":
    main()
