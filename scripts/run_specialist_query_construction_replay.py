#!/usr/bin/env python3
"""Replay specialist query constructions with the frozen ATR2 HPO winner."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_specialist_query_hpo import _load, _development_splits, _store_rows, _rank_frame
from scripts.run_frozen_specialist_query_hpo import TARGETS, DEFAULT_CONTRACT, DEFAULT_QUERY_POP
from scripts.run_frozen_specialist_query_hpo import SEED
OUT=ROOT/'data_perp/artifacts/frozen_specialist_query_construction_20260810_v1'
PARAMS={"n_estimators":180,"learning_rate":.03,"max_depth":4,"num_leaves":16,"min_child_samples":776,"min_sum_hessian_in_leaf":28.08104242513115,"min_gain_to_split":.003334820113493497,"colsample_bytree":.8397283415952219,"subsample":.7300957284014843,"subsample_freq":1,"reg_alpha":.0001226082411532739,"reg_lambda":1.745657954814456,"max_bin":127,"label_gain":[0.,.1,1.,3.,7.,12.],"verbosity":-1,"random_state":SEED,"n_jobs":1}
def qid(frame,mode):
 ts=pd.to_datetime(frame['__ts__'],utc=True)
 floor={'timestamp_side':None,'q1h_side':'1h','q4h_side':'4h'}[mode]
 key=ts if floor is None else ts.dt.floor(floor)
 return (key.astype('int64').astype(str)+'|'+frame.side_name.astype(str)).astype('string')
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True); frame,views=_load(DEFAULT_CONTRACT,DEFAULT_QUERY_POP); tr,va=_development_splits(frame); rows=[]; target='grade_atr_spacing_2p0'
 for mode in ('timestamp_side','q1h_side','q4h_side'):
  chunks=[]
  for side in ('long','short'):
   a,b=tr[tr.side_name.eq(side)].copy(),va[va.side_name.eq(side)].copy(); fields=sorted({f for v in views[side].values() for f in v});
   a=a.merge(_store_rows(a,fields),on='candidate_id',validate='one_to_one'); b=b.merge(_store_rows(b,fields),on='candidate_id',validate='one_to_one')
   a['query_id']=qid(a,mode)
   model,used,_=_rank_frame(a,fields,target,query_column='query_id',params=PARAMS)
   if mode!='q4h_side':
    # Refit with the alternative query; _rank_frame above is only used for its
    # stable sorting contract.
    x=a[['candidate_id','query_id',*fields,target]].copy(); x['__row__']=np.arange(len(x)); x=x.sort_values(['query_id','candidate_id'],kind='stable'); sizes=x.groupby('query_id',sort=False).size(); x=x[x.query_id.isin(sizes.index[sizes.ge(2)])]; groups=x.groupby('query_id',sort=False).size().to_numpy(np.int32); model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',lambdarank_truncation_level=10,**PARAMS); med=a[fields].apply(pd.to_numeric,errors='coerce').median(); y=a[target].to_numpy(float)[x['__row__'].to_numpy()]; model.fit(x[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.),y,group=groups); used=fields
   med=a[used].apply(pd.to_numeric,errors='coerce').median(); score=model.predict(b[used].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.)); z=b[['candidate_id','__ts__','side_name','net_bps','gross_bps']].copy(); z['score']=score; chunks.append(z)
  pred=pd.concat(chunks,ignore_index=True); rows.append({'query':mode,**global_tail_metrics(pred),**monthly_stability(pred)})
 pd.DataFrame(rows).to_parquet(out/'query_construction_metrics.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'frozen_specialist_query_construction_replay_v1','target':target,'params':PARAMS,'queries':['timestamp_side','q1h_side','q4h_side'],'selection':'global top5 net, then monthly stability, then top1 net'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); print(run(a.out))
