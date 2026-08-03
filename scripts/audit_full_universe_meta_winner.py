#!/usr/bin/env python3
"""Audit a fixed globally-ranked residual + reliability configuration."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def rank01(x:pd.Series)->pd.Series:return x.rank(method="average",pct=True)
def metrics(z:pd.DataFrame)->dict:
    return {"n":int(len(z)),"gross_bps":float(z.gross_bps.mean()),"net_bps":float(z.net_bps.mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())}
def select(z:pd.DataFrame,score:str,f:float)->pd.DataFrame:return z.sort_values([score,"candidate_id"],ascending=[False,True]).head(int(np.ceil(len(z)*f)))

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--reliability",type=Path,required=True);p.add_argument("--value",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--weight",type=float,default=.25);a=p.parse_args()
    r=pd.read_parquet(a.reliability);v=pd.read_parquet(a.value,columns=["candidate_id","final_score"]).rename(columns={"final_score":"value_score"})
    d=r.merge(v,on="candidate_id",validate="one_to_one");d["winner_score"]=(1-a.weight)*rank01(d.value_score)+a.weight*rank01(d.meta_score);d["base_score"]=d.base_expected_net_bps;d["month"]=pd.to_datetime(d.__ts__,utc=True).dt.strftime("%Y-%m");d["week"]=pd.to_datetime(d.__ts__,utc=True).dt.to_period("W").astype(str)
    result={"schema":"full_universe_winner_audit_v1","configuration":{"value":"residual around prequential expected net","reliability":"shared cost-clear probability","combination":f"{1-a.weight:.2f} value rank + {a.weight:.2f} reliability rank","selection":"global pooled across both sides and timestamps"},"global":{},"monthly_global_top10":{},"weekly_paired_top10":[],"selected_side_contributions":{}}
    for f in (.01,.05,.10,.20):
        w=select(d,"winner_score",f);b=select(d,"base_score",f);result["global"][str(f)]={"winner":metrics(w),"base":metrics(b),"net_lift_bps":float(w.net_bps.mean()-b.net_bps.mean())}
    for month,z in d.groupby("month",sort=True):result["monthly_global_top10"][month]={"winner":metrics(select(z,"winner_score",.1)),"base":metrics(select(z,"base_score",.1))}
    for week,z in d.groupby("week",sort=True):
        w,b=select(z,"winner_score",.1),select(z,"base_score",.1);result["weekly_paired_top10"].append({"week":week,"winner_net_bps":float(w.net_bps.mean()),"base_net_bps":float(b.net_bps.mean()),"lift_bps":float(w.net_bps.mean()-b.net_bps.mean())})
    w10=select(d,"winner_score",.1)
    for side,z in w10.groupby("side_name",sort=True):result["selected_side_contributions"][side]=metrics(z)
    lifts=np.array([x["lift_bps"] for x in result["weekly_paired_top10"]]);result["weekly_paired_summary"]={"n_weeks":len(lifts),"mean_lift_bps":float(lifts.mean()),"median_lift_bps":float(np.median(lifts)),"positive_weeks":int((lifts>0).sum()),"negative_weeks":int((lifts<0).sum())}
    a.out.mkdir(parents=True,exist_ok=True);d.to_parquet(a.out/"scored_predictions.parquet",index=False);(a.out/"audit.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
