#!/usr/bin/env python3
"""Causally fit a shrunk base-value × reliability net map on development.

The table is deliberately low-capacity.  It learns only on the development
period, fixes its quantile edges before OOS, leaves candidates outside a
selective reliability head untouched, and selects all table parameters on
development pooled-global top-k economics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _top(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> dict:
    z = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True]).head(int(np.ceil(len(frame) * fraction)))
    return {"n": int(len(z)), "gross_bps": float(z.gross_bps.mean()), "net_bps": float(z.net_bps.mean()), "long_n": int(z.side_name.eq("long").sum()), "short_n": int(z.side_name.eq("short").sum())}


def _edges(x: np.ndarray, bins: int) -> np.ndarray:
    bins = int(bins)
    edges = np.quantile(x, np.linspace(0., 1., bins + 1))
    # Identical quantiles are permitted in ill-conditioned samples; spread
    # them infinitesimally so bin assignment remains deterministic.
    return np.maximum.accumulate(edges + np.arange(len(edges)) * 1e-10)


def _index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2)


def _fit_score(dev: pd.DataFrame, apply: pd.DataFrame, bins: int, shrinkage: float, blend: float) -> tuple[np.ndarray, dict]:
    probability_col = "meta_score" if "meta_score" in dev else "reliability_score"
    apply_probability_col = "meta_score" if "meta_score" in apply else "reliability_score"
    eligible = dev[probability_col].notna().to_numpy()
    if eligible.sum() < bins * bins * 20:
        raise ValueError("not enough admitted candidates for requested table")
    base_edges = _edges(dev.loc[eligible, "value_score"].to_numpy(float), bins)
    reliability_edges = _edges(dev.loc[eligible, probability_col].to_numpy(float), bins)
    bi = _index(dev.value_score.to_numpy(float), base_edges)
    ri = _index(dev[probability_col].fillna(0.).to_numpy(float), reliability_edges)
    prior = float(dev.loc[eligible, "net_bps"].mean())
    sums = np.zeros((bins, bins)); counts = np.zeros((bins, bins))
    np.add.at(sums, (bi[eligible], ri[eligible]), dev.loc[eligible, "net_bps"].to_numpy(float))
    np.add.at(counts, (bi[eligible], ri[eligible]), 1.)
    mapped = (sums + shrinkage * prior) / (counts + shrinkage)
    eligible_apply = apply[apply_probability_col].notna().to_numpy()
    abi = _index(apply.value_score.to_numpy(float), base_edges)
    ari = _index(apply[apply_probability_col].fillna(0.).to_numpy(float), reliability_edges)
    score = apply.value_score.to_numpy(float).copy()
    # The table supplies an economically scaled, shrunk target only where the
    # reliability head was valid; blend=0 exactly recovers the value ranking.
    score[eligible_apply] = (1. - blend) * score[eligible_apply] + blend * mapped[abi[eligible_apply], ari[eligible_apply]]
    contract = {"bins": bins, "shrinkage_rows": shrinkage, "blend": blend, "prior_net_bps": prior, "base_edges": base_edges.tolist(), "reliability_edges": reliability_edges.tolist(), "cell_counts": counts.astype(int).tolist()}
    return score, contract


def _load(reliability: Path, value: Path) -> pd.DataFrame:
    r = pd.read_parquet(reliability)
    v = pd.read_parquet(value, columns=["candidate_id", "final_score"]).rename(columns={"final_score": "value_score"})
    return r.merge(v, on="candidate_id", validate="one_to_one")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-reliability", type=Path, required=True);p.add_argument("--oos-reliability", type=Path, required=True)
    p.add_argument("--development-value", type=Path, required=True);p.add_argument("--oos-value", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True);p.add_argument("--bins", default="3,5");p.add_argument("--shrinkage", default="250,1000,4000");p.add_argument("--blends", default="0,0.25,0.5,0.75,1");p.add_argument("--top-fraction", type=float, default=.10)
    a = p.parse_args();dev = _load(a.development_reliability, a.development_value);oos = _load(a.oos_reliability, a.oos_value)
    candidates=[]
    for bins in map(int, a.bins.split(",")):
        for shrinkage in map(float, a.shrinkage.split(",")):
            for blend in map(float, a.blends.split(",")):
                dev_score, contract = _fit_score(dev, dev, bins, shrinkage, blend)
                oos_score, _ = _fit_score(dev, oos, bins, shrinkage, blend)
                candidates.append({"contract": contract, "development": _top(dev, dev_score, a.top_fraction), "oos": _top(oos, oos_score, a.top_fraction)})
    selected = sorted(candidates, key=lambda x: (-x["development"]["net_bps"], -x["development"]["gross_bps"], x["contract"]["bins"], x["contract"]["shrinkage_rows"], x["contract"]["blend"]))[0]
    result={"schema":"full_universe_two_dimensional_trust_map_v1","selection":"development pooled-global top-k net; ties gross then lower complexity","selected":selected,"grid":candidates}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps({"selected":selected}))


if __name__ == "__main__":main()
