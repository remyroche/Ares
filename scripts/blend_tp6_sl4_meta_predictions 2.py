#!/usr/bin/env python3
"""Causally blend already side-calibrated TP6/SL4 meta candidates (MC4).

Each input must have been fit/calibrated independently per side on the same
chronological split.  The blend deliberately happens *after* that calibration:
every component is in expected-net bps on a comparable side-local scale, and
only then is a single global top-k book constructed.  This avoids a hidden
per-side quota and never refits on evaluation outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOPS = (.005, .01, .02, .03, .05, .10)
KEYS = ["candidate_id", "__ts__", "side_name"]


def _read(spec: str) -> tuple[str, pd.DataFrame]:
    name, value = spec.split("=", 1)
    path = Path(value) / "predictions.parquet"
    frame = pd.read_parquet(path)
    needed = KEYS + ["t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "side_calibrated_score_bps"]
    missing = set(needed) - set(frame)
    if missing:
        raise KeyError(f"{name} lacks {sorted(missing)}")
    return name, frame[needed].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="name=artifact directory; repeat")
    parser.add_argument("--weights", required=True, help="JSON object by input name; weights must sum positive")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    weights = {str(k): float(v) for k, v in json.loads(args.weights).items()}
    if min(weights.values(), default=-1.) < 0. or sum(weights.values()) <= 0.:
        raise ValueError("blend weights must be non-negative and sum positive")
    components = [_read(spec) for spec in args.input]
    names = [name for name, _ in components]
    if set(names) != set(weights):
        raise ValueError("weights must name every input exactly once")
    base_name, combined = components[0]
    combined = combined.rename(columns={"side_calibrated_score_bps": f"score_{base_name}"})
    for name, frame in components[1:]:
        frame = frame[KEYS + ["side_calibrated_score_bps"]].rename(columns={"side_calibrated_score_bps": f"score_{name}"})
        combined = combined.merge(frame, on=KEYS, how="inner", validate="one_to_one")
    if len(combined) != len(components[0][1]):
        raise ValueError("MC4 inputs do not cover the same evaluation population")
    combined["side_calibrated_score_bps"] = sum(weights[name] * combined[f"score_{name}"] for name in names) / sum(weights.values())
    # Component calibrated scores have already satisfied the per-side
    # calibration requirement.  Raw score is only a deterministic tie-break.
    combined["meta_raw_score"] = combined["side_calibrated_score_bps"]
    rank = combined.sort_values(["side_calibrated_score_bps", "meta_raw_score", "candidate_id"], ascending=[False, False, True], kind="mergesort")
    metrics: list[dict[str, object]] = []
    for top in TOPS:
        selected = rank.head(int(np.ceil(top * len(rank))))
        for side, view in (("global", selected), ("long", selected[selected.side_name.eq("long")]), ("short", selected[selected.side_name.eq("short")])):
            metrics.append({"allocation": "global_after_side_calibration", "attribution_side": side, "top_fraction": top, "n": len(view), "gross_bps": float(view.t4_tp6_sl4_gross_bps.mean()), "net_bps": float(view.t4_tp6_sl4_net_bps.mean())})
        for side in ("long", "short"):
            view = combined[combined.side_name.eq(side)].sort_values(["side_calibrated_score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(int(np.ceil(top * (combined.side_name == side).sum())))
            metrics.append({"allocation": "per_side", "attribution_side": side, "top_fraction": top, "n": len(view), "gross_bps": float(view.t4_tp6_sl4_gross_bps.mean()), "net_bps": float(view.t4_tp6_sl4_net_bps.mean())})
    args.out.mkdir(parents=True)
    combined.to_parquet(args.out / "predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(args.out / "metrics.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({"schema": "tp6_sl4_mc4_side_calibrated_blend_v1", "inputs": {name: str(spec) for name, spec in zip(names, args.input)}, "weights": weights, "invariant": "per-side calibration before unconstrained global allocation", "metrics": metrics}, indent=2) + "\n")


if __name__ == "__main__":
    main()
