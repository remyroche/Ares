#!/usr/bin/env python3
"""Development-select a two-part failure penalty, then replay it OOS."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def metric(d,score,f):
    z=d.assign(_score=score).sort_values(["_score","candidate_id"],ascending=[False,True]).head(int(np.ceil(len(d)*f)))
    return {"n":int(len(z)),"gross_bps":float(z.gross_bps.mean()),"net_bps":float(z.net_bps.mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())}
def load(failure:Path,severity:Path,value:Path|None):
    f=pd.read_parquet(failure);s=pd.read_parquet(severity,columns=["candidate_id","final_score"]).rename(columns={"final_score":"severity_bps"});d=f.merge(s,on="candidate_id",validate="one_to_one")
    if value:
        v=pd.read_parquet(value,columns=["candidate_id","final_score"]).rename(columns={"final_score":"value_score"});d=d.merge(v,on="candidate_id",validate="one_to_one")
    else:d["value_score"]=d.base_expected_net_bps
    d["failure_loss_bps"]=d.meta_score*d.severity_bps
    return d
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--development-failure",type=Path,required=True);p.add_argument("--development-severity",type=Path,required=True);p.add_argument("--oos-failure",type=Path,required=True);p.add_argument("--oos-severity",type=Path,required=True);p.add_argument("--development-value",type=Path);p.add_argument("--oos-value",type=Path);p.add_argument("--out",type=Path,required=True);p.add_argument("--gammas",default="0,0.125,0.25,0.5,1,1.5,2");p.add_argument("--top-fraction",type=float,default=.10);a=p.parse_args()
    if bool(a.development_value)!=bool(a.oos_value):raise ValueError("provide both value files or neither")
    dev=load(a.development_failure,a.development_severity,a.development_value);oos=load(a.oos_failure,a.oos_severity,a.oos_value);rows=[]
    for gamma in map(float,a.gammas.split(',')):
        rows.append({"gamma":gamma,"development":metric(dev,dev.value_score-gamma*dev.failure_loss_bps,a.top_fraction),"oos":metric(oos,oos.value_score-gamma*oos.failure_loss_bps,a.top_fraction)})
    selected=sorted(rows,key=lambda x:(-x["development"]["net_bps"],-x["development"]["gross_bps"],x["gamma"]))[0]
    result={"schema":"full_universe_downside_penalty_v1","value":"residual-adjusted expected net" if a.development_value else "base expected net","penalty":"P(failure) times E(-net | failure)","selection":"development pooled global top-k", "selected_gamma":selected["gamma"],"selected_development":selected["development"],"selected_oos":selected["oos"],"grid":rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
