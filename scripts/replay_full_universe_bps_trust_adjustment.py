#!/usr/bin/env python3
"""Development-select centered or log-odds reliability corrections in bps."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def metric(d,score,f):
    z=d.assign(_score=score).sort_values(["_score","candidate_id"],ascending=[False,True]).head(int(np.ceil(len(d)*f)))
    return {"n":int(len(z)),"gross_bps":float(z.gross_bps.mean()),"net_bps":float(z.net_bps.mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())}
def load(r,v):
    d=pd.read_parquet(r);v=pd.read_parquet(v,columns=["candidate_id","final_score"]).rename(columns={"final_score":"value_score"});return d.merge(v,on="candidate_id",validate="one_to_one")
def score(d,prior,lam,rule):
    p=d.reliability_score.to_numpy(float);base=d.value_score.to_numpy(float);ok=np.isfinite(p);out=base.copy()
    if rule=='centered': correction=p-prior
    else:
        q=np.clip(p,1e-4,1-1e-4);pq=np.clip(prior,1e-4,1-1e-4);correction=np.log(q/(1-q))-np.log(pq/(1-pq))
    out[ok]+=lam*correction[ok];return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-reliability',type=Path,required=True);p.add_argument('--oos-reliability',type=Path,required=True);p.add_argument('--development-value',type=Path,required=True);p.add_argument('--oos-value',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rule',choices=('centered','log_odds'),required=True);p.add_argument('--lambdas',default='0,25,50,100,150,200,300');p.add_argument('--top-fraction',type=float,default=.10);a=p.parse_args()
    dev=load(a.development_reliability,a.development_value);oos=load(a.oos_reliability,a.oos_value);prior=float(dev.reliability_score.mean(skipna=True));rows=[]
    for lam in map(float,a.lambdas.split(',')):rows.append({'lambda_bps':lam,'development':metric(dev,score(dev,prior,lam,a.rule),a.top_fraction),'oos':metric(oos,score(oos,prior,lam,a.rule),a.top_fraction)})
    selected=sorted(rows,key=lambda x:(-x['development']['net_bps'],-x['development']['gross_bps'],x['lambda_bps']))[0];result={'schema':'full_universe_bps_trust_adjustment_v1','rule':a.rule,'target_prior':prior,'selection':'development pooled-global top-k net','selected_lambda_bps':selected['lambda_bps'],'selected_development':selected['development'],'selected_oos':selected['oos'],'grid':rows};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
