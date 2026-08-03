#!/usr/bin/env python3
"""Replay frozen Round-1 selections under the reference H12 triple barrier.

The score and global selections are never changed.  A first favourable hit
realises the frozen economic upper barrier; a first adverse hit realises the
frozen one-ATR lower barrier (adverse precedence on a same-minute conflict);
timeouts realise the H12 endpoint.  The already-frozen candidate row cost is
charged once, so the result is a matched-cost *simple exit-policy* diagnostic,
not a claim about a newly optimised execution cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.candidate_evaluation import stable_global_top_k

FRACTIONS = (.0025, .005, .01, .02, .05, .10, .20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--supportive-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = pd.read_parquet(args.predictions)
    fields = ["candidate_id", "economic_upper_return", "adverse_lower_return", "endpoint_signed_return", "clean_economic_favorable_first", "adverse_first", "timeout"]
    support = pd.read_parquet(args.supportive_labels, columns=fields)
    work = score.merge(support, on="candidate_id", validate="many_to_one", how="left")
    if work[fields[1:]].isna().any().any():
        raise ValueError("missing simple triple-barrier exit support")
    clean = work.clean_economic_favorable_first.astype(bool).to_numpy()
    adverse = work.adverse_first.astype(bool).to_numpy()
    timeout = work.timeout.astype(bool).to_numpy()
    if not np.array_equal(clean.astype(int) + adverse.astype(int) + timeout.astype(int), np.ones(len(work), dtype=int)):
        raise ValueError("reference barrier events are not exhaustive/mutually exclusive")
    gross = np.select([clean, adverse, timeout], [work.economic_upper_return, -work.adverse_lower_return, work.endpoint_signed_return]).astype(float)
    work["simple_tb_gross_return"] = gross
    work["simple_tb_net_return"] = gross - work.execution_cost_return.to_numpy(float)
    rows = []
    attr = []
    for (arm, variant), group in work.groupby(["target_arm", "model_variant"], observed=True, sort=True):
        for fraction in FRACTIONS:
            selected = stable_global_top_k(group, "score_bps", fraction)
            rows.append({"target_arm": arm, "model_variant": variant, "top_fraction": fraction, "selected_rows": int(len(selected)),
                         "simple_tb_gross_bps": float(selected.simple_tb_gross_return.mean() * 10_000),
                         "simple_tb_cost_bps": float(selected.execution_cost_return.mean() * 10_000),
                         "simple_tb_net_bps": float(selected.simple_tb_net_return.mean() * 10_000),
                         "clean_first_rate": float(selected.clean_economic_favorable_first.mean()),
                         "adverse_first_rate": float(selected.adverse_first.mean()),
                         "timeout_rate": float(selected.timeout.mean())})
            for dim, col in (("side", "side_name"), ("month", "__ts__")):
                local = selected.copy()
                if dim == "month":
                    local["month"] = pd.to_datetime(local[col], utc=True).dt.to_period("M").astype(str); col = "month"
                for value, part in local.groupby(col, observed=True, sort=True):
                    attr.append({"target_arm": arm, "model_variant": variant, "top_fraction": fraction, "dimension": dim, "value": str(value), "selected_rows": int(len(part)), "simple_tb_gross_bps": float(part.simple_tb_gross_return.mean()*10_000), "simple_tb_net_bps": float(part.simple_tb_net_return.mean()*10_000)})
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(args.output / "simple_triple_barrier_exit_results.parquet", index=False)
    pd.DataFrame(attr).to_parquet(args.output / "simple_triple_barrier_exit_attribution.parquet", index=False)
    (args.output / "manifest.json").write_text(json.dumps({"schema": "round1_simple_triple_barrier_exit_v1", "policy": "upper=max(1.5ATR,1.5%,row_cost); lower=1ATR; adverse tie precedence; timeout=H12 endpoint; frozen row cost charged once", "selection": "same independently pooled-global score ranking as Round 1"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
