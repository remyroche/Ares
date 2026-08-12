#!/usr/bin/env python3
"""Materialize deterministic Top-1 transitions for all real attribution arms."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("full_current", "gam_residual_only", "gam_plus_valid", "transport_only")


def _one(pred: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = ["candidate_id", "month", "stack_score", "exact_net_bps", "exact_gross_bps", "base_expected_bps", "gam_delta_bps", "gam_residual_bps", "gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count"]
    c = pred.loc[pred.arm.eq("control"), fields].rename(columns={"stack_score": "control_score"})
    a = pred.loc[pred.arm.eq(arm), fields].rename(columns={"stack_score": "arm_score"})
    m = c.merge(a[["candidate_id", "month", "arm_score"]], on=["candidate_id", "month"], validate="one_to_one")
    n = max(1, int(np.ceil(len(m) * 0.01)))
    cids = set(m.sort_values(["control_score", "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    aids = set(m.sort_values(["arm_score", "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    m["transition"] = np.where(m.candidate_id.isin(aids) & ~m.candidate_id.isin(cids), "entered_top1", np.where(m.candidate_id.isin(cids) & ~m.candidate_id.isin(aids), "exited_top1", np.where(m.candidate_id.isin(cids) & m.candidate_id.isin(aids), "stayed_top1", "outside_top1")))
    m["arm"] = arm
    summary = m.groupby(["arm", "transition"], sort=True).agg(rows=("candidate_id", "size"), mean_net_bps=("exact_net_bps", "mean"), median_net_bps=("exact_net_bps", "median"), mean_base_expected_bps=("base_expected_bps", "mean"), mean_gam_delta_bps=("gam_delta_bps", "mean"), mean_gam_residual_bps=("gam_residual_bps", "mean"), valid_fraction=("gam_transport_valid", "mean"), mean_matched_mass=("gam_matched_mass", "mean"), mean_unmatched_mass=("gam_unmatched_mass", "mean"), mean_archetype_count=("gam_archetype_count", "mean"), mean_cluster_count=("gam_cluster_count", "mean"), mean_control_score=("control_score", "mean"), mean_arm_score=("arm_score", "mean")).reset_index()
    monthly = m.loc[m.transition.isin(["entered_top1", "exited_top1"])].groupby(["arm", "month", "transition"], sort=True).agg(rows=("candidate_id", "size"), mean_net_bps=("exact_net_bps", "mean"), mean_gam_delta_bps=("gam_delta_bps", "mean"), valid_fraction=("gam_transport_valid", "mean")).reset_index()
    return summary, monthly


def run(input_dir: Path) -> None:
    pred = pd.read_parquet(input_dir / "predictions.parquet")
    summaries, monthly = [], []
    for arm in ARMS:
        s, m = _one(pred, arm); summaries.append(s); monthly.append(m)
    pd.concat(summaries, ignore_index=True).to_parquet(input_dir / "top1_transition_summary_all.parquet", index=False)
    pd.concat(monthly, ignore_index=True).to_parquet(input_dir / "top1_transition_monthly_all.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    run(parser.parse_args().input_dir)
