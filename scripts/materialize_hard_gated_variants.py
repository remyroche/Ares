#!/usr/bin/env python3
"""Materialize month-valid hard-gated scores for the decomposition arms."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def run(input_dir: Path) -> None:
    p = pd.read_parquet(input_dir / "predictions_decomposition.parquet")
    p["stack_score"] = p.groupby(["arm", "month"], sort=False)["stack_score"].transform(lambda z: z.rank(pct=True, method="average"))
    base = p[p.arm.eq("control")][["candidate_id", "month", "stack_score", "exact_net_bps", "exact_gross_bps", "gam_month_valid"]].rename(columns={"stack_score": "control"})
    arms = ["gam_delta_only", "gam_residual_only", "gam_delta_residual", "gam_delta_residual_valid"]
    rows = []
    for arm in arms:
        a = p[p.arm.eq(arm)][["candidate_id", "month", "stack_score"]].rename(columns={"stack_score": "enhanced"})
        m = base.merge(a, on=["candidate_id", "month"], validate="one_to_one")
        m["gated"] = np.where(m.gam_month_valid.eq(1), m.enhanced, m.control)
        for score in ["control", "enhanced", "gated"]:
            for tail in (0.005, 0.01, 0.02, 0.05, 0.10):
                n = max(1, int(math.ceil(len(m) * tail))); top = m.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                rows.append({"arm": arm, "score": score, "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean())})
    pd.DataFrame(rows).to_parquet(input_dir / "metrics_hard_gated_variants.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path, required=True); run(parser.parse_args().input_dir)
