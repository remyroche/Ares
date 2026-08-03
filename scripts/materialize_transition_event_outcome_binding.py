#!/usr/bin/env python3
"""Bind transition events to exact candidate outcomes by post-event 24h windows."""
from __future__ import annotations
import hashlib,json,os,uuid
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'data_perp/artifacts'; OUT=A/'transition_event_outcome_binding_20260730_v1'
def h(p):
 d=hashlib.sha256();d.update(p.read_bytes());return d.hexdigest()
def run(out=OUT):
 if out.exists():raise FileExistsError(out)
 events=pd.read_parquet(A/'transition_pattern_catalogue_20260730_v6/event_preonset_sequences.parquet',columns=['event_id','anchor_source_utc']);events['anchor_source_utc']=pd.to_datetime(events.anchor_source_utc,utc=True);events['window_end']=events.anchor_source_utc+pd.Timedelta(hours=24)
 sources=[('A_2022_23',A/'failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet'),('A_2024',A/'failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet'),('B_2025',A/'febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet'),('B_2026',A/'mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/full_ic.parquet')]
 bound=[]; unmatched=[]
 for grade,p in sources:
  x=pd.read_parquet(p); ts=next((c for c in ['__ts__','execution_decision_utc','timestamp'] if c in x),None)
  if ts is None:unmatched.append({'source':grade,'reason':'no candidate timestamp','rows':len(x)});continue
  x['_ts']=pd.to_datetime(x[ts],utc=True,errors='coerce'); cols=[c for c in ['execution_net_ev_12h','execution_gross_ev_12h','__opportunity_occurred_12h__','__peak_mfe_atr_12h__','execution_mae_return_12h'] if c in x]
  for _,e in events.iterrows():
   q=x[(x._ts>=e.anchor_source_utc)&(x._ts<e.window_end)]
   if len(q): bound.append({'event_id':e.event_id,'source_grade':grade,'candidate_rows':len(q),**{c:float(q[c].mean()) for c in cols}})

 b=pd.DataFrame(bound); coverage=events[['event_id']].merge(b.groupby('event_id').candidate_rows.sum().rename('rows'),on='event_id',how='left');coverage['matched']=coverage.rows.notna();coverage['unmatched_reason']=coverage.matched.map({True:'',False:'no exact candidate outcome in supplied source windows'})
 stage=out.parent/f'.{out.name}.{uuid.uuid4().hex}';stage.mkdir();b.to_parquet(stage/'event_outcomes.parquet',index=False);coverage.to_csv(stage/'coverage.csv',index=False);pd.DataFrame(unmatched).to_csv(stage/'source_skips.csv',index=False);m={'schema':'transition_event_outcome_binding_v1','status':'PARTIAL_EXACT_EVALUATION_ONLY','promotion_eligible':False,'window':'[event anchor,event anchor+24h)','post_event_labels_only':True,'sources':{str(p):h(p) for _,p in sources},'outputs':{'event_outcomes':len(b),'coverage':len(coverage)}};(stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(stage/'manifest.sha256').write_text(h(stage/'manifest.json')+'  manifest.json\n');os.replace(stage,out);return m
if __name__=='__main__':print(json.dumps(run()))
