#!/usr/bin/env python3
"""Summarize a narrow R5 gate ledger on the matched frozen outcome ledger."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TAILS = (0.01, 0.02, 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--selection-ledger", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    manifest = json.loads(args.selection_manifest.read_text())
    contracts = {
        arm: (values["expected"], values["admitted"])
        for arm, values in manifest["contracts"].items()
    }
    outcomes = pd.read_parquet(args.outcome_ledger, columns=[
        "candidate_id", "__decision_ts__", "final_score", "policy_path_valid",
        "policy_net_bps",
    ])
    selection = pd.read_parquet(args.selection_ledger)
    for frame in (outcomes, selection):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if not outcomes[["candidate_id", "__decision_ts__"]].equals(
        selection[["candidate_id", "__decision_ts__"]]
    ):
        raise ValueError("selection and outcome ledgers do not share identical ordered identities")
    core = outcomes.copy()
    rows = []
    monthly = []
    for arm, (expected, admitted) in contracts.items():
        mask = selection[admitted].fillna(False).astype(bool).to_numpy()
        pool = core.loc[mask].assign(
            expected=pd.to_numeric(selection.loc[mask, expected], errors="coerce").to_numpy(float),
        ).sort_values(
            ["expected", "final_score", "candidate_id"],
            ascending=[False, False, True], kind="stable",
        )
        choices = [("all_admitted", pool)]
        choices.extend((
            f"top_{tail:g}", pool.head(max(1, int(math.ceil(tail * len(pool)))))
        ) for tail in TAILS)
        for kind, chosen in choices:
            valid = chosen.loc[
                chosen["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(chosen["policy_net_bps"], errors="coerce"))
            ]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "kind": kind, "admitted_rows": int(len(pool)),
                "selected_rows": int(len(chosen)), "valid_outcomes": int(len(valid)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0).mean()) if len(net) else np.nan,
            })
        pool["month"] = pool["__decision_ts__"].dt.strftime("%Y-%m")
        for month, group in pool.groupby("month", sort=True):
            valid = group.loc[group["policy_path_valid"].fillna(False).astype(bool)]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce").dropna()
            monthly.append({
                "arm": arm, "month": month, "admitted_rows": int(len(group)),
                "valid_outcomes": int(len(net)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
            })
    args.out_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(args.out_dir / "raw_metrics.parquet", index=False)
    pd.DataFrame(monthly).to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "r5_domain_probability_summary_v1",
        "outcome_ledger": str(args.outcome_ledger),
        "selection_ledger": str(args.selection_ledger),
        "contracts": manifest["contracts"],
        "winner_promoted": False,
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "arms": len(contracts), "rows": len(core)}))


if __name__ == "__main__":
    main()
