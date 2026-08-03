#!/usr/bin/env python3
"""Choose a monotone meta correction on a development period, then replay it OOS.

Inputs are the materialised outputs of run_full_universe_round_b_meta_targets.
Selection is global (both sides and all timestamps pooled), and chooses only a
single predeclared correction magnitude.  It never learns from the OOS file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> dict:
    ranked = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True])
    top = ranked.head(int(np.ceil(len(ranked) * fraction)))
    return {
        "n": int(len(top)),
        "gross_bps": float(top.gross_bps.mean()),
        "net_bps": float(top.net_bps.mean()),
        "long_n": int(top.side_name.eq("long").sum()),
        "short_n": int(top.side_name.eq("short").sum()),
    }


def _score(frame: pd.DataFrame, rule: str, strength: float) -> np.ndarray:
    base = frame.base_expected_net_bps.to_numpy(float)
    meta = frame.meta_score.to_numpy(float)
    if np.isnan(meta).any():
        raise ValueError("combination inputs must be all-candidate (no population filtering)")
    if rule == "risk_penalty":
        return base - strength * meta
    if rule == "correctness_bonus":
        return base + strength * meta
    if rule == "residual_add":
        # final_score already incorporates the fitted residual plus its
        # train-only centring correction; scale that correction, not the base.
        return base + strength * (frame.final_score.to_numpy(float) - base)
    raise ValueError(rule)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--oos", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rule", choices=("risk_penalty", "correctness_bonus", "residual_add"), required=True)
    p.add_argument("--top-fraction", type=float, default=.10)
    p.add_argument("--strengths", default="0,25,50,100,150,200,300")
    a = p.parse_args()
    dev = pd.read_parquet(a.development)
    oos = pd.read_parquet(a.oos)
    strengths = [float(x) for x in a.strengths.split(",")]
    rows = []
    for strength in strengths:
        rows.append({"strength": strength, "development": _metrics(dev, _score(dev, a.rule, strength), a.top_fraction)})
    # Deterministic selection: maximise net, then gross, then smaller correction.
    chosen = sorted(rows, key=lambda x: (-x["development"]["net_bps"], -x["development"]["gross_bps"], x["strength"]))[0]
    for row in rows:
        row["oos"] = _metrics(oos, _score(oos, a.rule, row["strength"]), a.top_fraction)
    selected = next(x for x in rows if x["strength"] == chosen["strength"])
    result = {"rule": a.rule, "top_fraction": a.top_fraction, "selection": "development pooled-global top-k net, tie gross then smaller strength", "selected_strength": chosen["strength"], "selected_development": chosen["development"], "selected_oos": selected["oos"], "grid": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
