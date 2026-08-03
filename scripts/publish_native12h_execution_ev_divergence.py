#!/usr/bin/env python3
"""Held-out exact execution-EV comparison after native OOF scores freeze."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SCORES=ROOT/'data_perp/artifacts/febapr2025_native12h_matched_score_divergence_20260729_v1/identical_rows.parquet';EV=ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet';OUT=ROOT/'data_perp/artifacts/febapr2025_native12h_execution_ev_divergence_20260729_v1'
def metric(x,s):
 k=max(1,int(len(x)*.1));p=x.nlargest(k,s);return {'rows':len(x),'top10_rows':k,'gross_ev_mean':float(p.execution_gross_ev_12h.mean()),'cost_mean':float(p.execution_cost_return.mean()),'net_ev_mean':float(p.execution_net_ev_12h.mean()),'positive_net_fraction':float((p.execution_net_ev_12h>0).mean()),'net_ev_sum':float(p.execution_net_ev_12h.sum())}
def main():
 x=pd.read_parquet(SCORES);e=pd.read_parquet(EV,columns=['candidate_id','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h']);x=x.merge(e,on='candidate_id',how='inner',validate='one_to_one');x['month']=pd.to_datetime(x.__ts__,utc=True).dt.strftime('%Y-%m');r={'schema':'native12h_retrain_heldout_execution_ev_divergence_v1','scope':'scores frozen before one-to-one signed execution-EV join','rows':len(x),'overall':{'old_24h_score':metric(x,'old_score'),'new_12h_score':metric(x,'new_score')},'by_month_side':[{**{'month':m,'side':s},'old_24h_score':metric(g,'old_score'),'new_12h_score':metric(g,'new_score')} for (m,s),g in x.groupby(['month','side_name'],sort=True)]};OUT.mkdir();x.to_parquet(OUT/'joined_scores_execution_ev.parquet',index=False,compression='zstd');(OUT/'report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
