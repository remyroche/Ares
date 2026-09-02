#!/usr/bin/env python3
"""Audit target oracle economics by small, decision-time causal regimes.

The labels use no regime information.  Regimes only stratify outcomes here,
using fields available at the candidate decision.  They are deliberately broad
and fixed-threshold, avoiding an ex-post cluster that would make a semantic
reversal look more precise than it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_FIELDS = ["atr_percentile", "trend_strength_percentile", "market_breadth_24h", "mkt_atr_expansion_4h"]


def _regimes(x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "vol_low": x.atr_percentile.le(-.67).to_numpy(),
        "vol_medium": x.atr_percentile.gt(-.67).to_numpy() & x.atr_percentile.lt(.67).to_numpy(),
        "vol_high": x.atr_percentile.ge(.67).to_numpy(),
        "trend_weak": x.trend_strength_percentile.le(-.67).to_numpy(),
        "trend_neutral": x.trend_strength_percentile.gt(-.67).to_numpy() & x.trend_strength_percentile.lt(.67).to_numpy(),
        "trend_strong": x.trend_strength_percentile.ge(.67).to_numpy(),
        "breadth_narrow": x.market_breadth_24h.le(.68).to_numpy(),
        "breadth_mid": x.market_breadth_24h.gt(.68).to_numpy() & x.market_breadth_24h.lt(.76).to_numpy(),
        "breadth_broad": x.market_breadth_24h.ge(.76).to_numpy(),
        "vol_contracting": x.mkt_atr_expansion_4h.le(.75).to_numpy(),
        "vol_transition": x.mkt_atr_expansion_4h.gt(.75).to_numpy() & x.mkt_atr_expansion_4h.lt(1.).to_numpy(),
        "vol_expanding": x.mkt_atr_expansion_4h.ge(1.).to_numpy(),
    }


def _deciles(g: pd.DataFrame, score: str) -> pd.Series:
    # rank(method=first) makes complete, deterministic deciles even if the
    # target saturates at its sigmoid tails.
    return pd.qcut(g[score].rank(method="first"), 10, labels=False) + 1


def _audit(g: pd.DataFrame, target: str, regime: str, side: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    g = g.copy(); g["decile"] = _deciles(g, target)
    rows=[]
    for d, z in g.groupby("decile", observed=True):
        rows.append({"target":target,"regime":regime,"side":side,"decile":int(d),"rows":len(z),"gross_bps":z.gross_bps.mean(),"net_bps":z.net_bps.mean(),"clear_rate":z.robust_clear_event_b25.mean()})
    means = pd.DataFrame(rows).sort_values("decile")
    top = g.nlargest(max(1, int(np.ceil(len(g)*.10))), target)
    monotonic = float(pd.Series(np.arange(1,11)).corr(means.net_bps, method="spearman")) if len(means) == 10 else np.nan
    summary={"target":target,"regime":regime,"side":side,"rows":len(g),"decile_net_spearman":monotonic,"oracle_top10_gross_bps":top.gross_bps.mean(),"oracle_top10_net_bps":top.net_bps.mean(),"lowest_decile_net_bps":means.iloc[0].net_bps,"highest_decile_net_bps":means.iloc[-1].net_bps,"semantic_reversal":bool(len(g)>=10000 and means.iloc[-1].net_bps <= means.iloc[0].net_bps),"support_insufficient":bool(len(g)<10000)}
    return rows, summary


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path,required=True); p.add_argument("--winner",type=Path,required=True); p.add_argument("--robust",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    pieces=[]
    identity=["candidate_id","side_name",*REGIME_FIELDS]
    winner_cols=["candidate_id","t4_tp6_sl4_gross_bps","t4_tp6_sl4_net_bps"]
    label_cols=["candidate_id","label_valid","lower_touch_minute","robust_clear_event_b25","robust_clear_soft_b25_t50"]
    for part in sorted((a.panel/"parts").glob("*.parquet")):
        base=pd.read_parquet(part,columns=identity)
        outcome=pd.read_parquet(a.winner/"parts"/part.name,columns=winner_cols)
        labels=pd.read_parquet(a.robust/"parts"/part.name,columns=label_cols)
        x=base.merge(outcome,on="candidate_id",validate="one_to_one").merge(labels,on="candidate_id",validate="one_to_one")
        x=x.loc[x.label_valid.eq(True)].rename(columns={"t4_tp6_sl4_gross_bps":"gross_bps","t4_tp6_sl4_net_bps":"net_bps"})
        pieces.append(x)
    x=pd.concat(pieces,ignore_index=True).dropna(subset=REGIME_FIELDS)
    x["R2_robust_clear_soft_b25_t50"]=x.robust_clear_soft_b25_t50
    x["R3_economic_simplex_b25"]=np.select([x.robust_clear_event_b25.eq(1),x.lower_touch_minute.ge(0)], [2.,0.], default=1.)
    decile_rows=[]; summaries=[]; masks={"all":np.ones(len(x),dtype=bool),**_regimes(x)}
    for regime, mask in masks.items():
        for side in ("long","short"):
            g=x.loc[mask & x.side_name.eq(side)]
            for target in ("R2_robust_clear_soft_b25_t50", "R3_economic_simplex_b25"):
                d,s=_audit(g,target,regime,side); decile_rows.extend(d); summaries.append(s)
    pd.DataFrame(decile_rows).to_parquet(a.out/"target_regime_decile_economics.parquet",index=False)
    pd.DataFrame(summaries).to_parquet(a.out/"target_regime_oracle_results.parquet",index=False)
    manifest={"schema":"tp6_target_regime_oracle_v1","contract":"TP6/SL4/H12, exact entry, 100bps cost","regime_fields":REGIME_FIELDS,"regime_thresholds":"fixed decision-time normalized values; diagnostic stratification only","targets":["R2 robust clear b25 t50","R3 coarse economic proxy"],"rows":len(x)}
    (a.out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__': main()
