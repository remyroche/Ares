#!/usr/bin/env python3
"""Development-select simple causal meta action rules and replay them untouched.

The base rank is always common-unit expected net and selection is one global
pool.  ``cost_clear`` is a probability head; it is never ranked as an upside
score by itself.  Parameters/cutoffs are selected only on the supplied
development predictions and carried unchanged to the OOS predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(float)


def _top(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> dict:
    z = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True]).head(int(np.ceil(len(frame) * fraction)))
    return {"n": int(len(z)), "gross_bps": float(z.gross_bps.mean()), "net_bps": float(z.net_bps.mean()), "long_n": int(z.side_name.eq("long").sum()), "short_n": int(z.side_name.eq("short").sum())}


def _score(frame: pd.DataFrame, rule: str, parameter: float, prob_cutoff: float) -> np.ndarray:
    base = frame.base_score.to_numpy(float)
    probability_col = "meta_score" if "meta_score" in frame else "reliability_score"
    p = frame[probability_col].to_numpy(float)
    eligible = np.isfinite(p)
    if rule == "rank_blend":
        base_rank = _rank01(base)
        # Selective models have no score outside their causal high-base
        # population.  Those rows retain exactly their base rank.
        reliability_rank = base_rank.copy()
        reliability_rank[eligible] = _rank01(p[eligible])
        return (1. - parameter) * base_rank + parameter * reliability_rank
    if rule == "multiplicative_trust":
        # No change to non-positive opportunities: probability is a trust
        # multiplier, not a way to turn a negative expected trade positive.
        # ``parameter`` interpolates from no overlay (0) to full trust (1).
        return np.where(eligible & (base > 0.), base * ((1. - parameter) + parameter * p), base)
    if rule == "veto":
        # The fixed probability cutoff comes entirely from development.  A
        # rejected candidate cannot enter the global top-k book.
        return np.where(~eligible | (p >= prob_cutoff), base, -np.inf)
    raise ValueError(rule)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development", type=Path, required=True)
    ap.add_argument("--oos", type=Path, required=True)
    ap.add_argument("--development-value",type=Path,help="optional residual/value predictions keyed by candidate_id")
    ap.add_argument("--oos-value",type=Path,help="optional residual/value predictions keyed by candidate_id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rule", choices=("rank_blend", "multiplicative_trust", "veto"), required=True)
    ap.add_argument("--top-fraction", type=float, default=.10)
    ap.add_argument("--parameters", default="0,0.25,0.5,0.75")
    a = ap.parse_args()
    dev, oos = pd.read_parquet(a.development), pd.read_parquet(a.oos)
    if bool(a.development_value) != bool(a.oos_value):
        raise ValueError("provide both value files or neither")
    if a.development_value:
        for name, frame, path in (("development",dev,a.development_value),("oos",oos,a.oos_value)):
            values=pd.read_parquet(path,columns=["candidate_id","final_score"]).rename(columns={"final_score":"base_score"})
            frame.merge(values,on="candidate_id",validate="one_to_one",how="left",copy=False)
            frame["base_score"]=frame.candidate_id.map(values.set_index("candidate_id").base_score)
            if frame.base_score.isna().any():raise ValueError(f"{name} value file does not cover every candidate")
    else:
        dev["base_score"]=dev.base_expected_net_bps;oos["base_score"]=oos.base_expected_net_bps
    params = [float(x) for x in a.parameters.split(",")]
    candidates=[]
    for param in params:
        # Veto parameter is the fraction to reject.  The probability cutoff is
        # frozen on development, then applied verbatim OOS.
        probability_col = "meta_score" if "meta_score" in dev else "reliability_score"
        cutoff = float(dev.loc[dev[probability_col].notna(), probability_col].quantile(param)) if a.rule == "veto" else float("nan")
        candidates.append({"parameter": param, "probability_cutoff": cutoff, "development": _top(dev, _score(dev, a.rule, param, cutoff), a.top_fraction)})
    selected=sorted(candidates,key=lambda r:(-r["development"]["net_bps"],-r["development"]["gross_bps"],r["parameter"]))[0]
    for r in candidates:
        r["oos"]=_top(oos,_score(oos,a.rule,r["parameter"],r["probability_cutoff"]),a.top_fraction)
    chosen=next(r for r in candidates if r["parameter"]==selected["parameter"])
    result={"rule":a.rule,"value_representation":"residual-adjusted expected net" if a.development_value else "base expected net","top_fraction":a.top_fraction,"selection":"development pooled-global top-k net; tie gross then smaller action", "selected_parameter":chosen["parameter"],"selected_probability_cutoff":chosen["probability_cutoff"],"selected_development":chosen["development"],"selected_oos":chosen["oos"],"grid":candidates}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__ == "__main__":
    main()
