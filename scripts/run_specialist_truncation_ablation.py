#!/usr/bin/env python3
"""Bounded LambdaRank truncation-level ablation for the ATR2 specialist winner."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_specialist_query_hpo import _load, _development_splits, _store_rows, _query_id, SEED
OUT=ROOT/'data_perp/artifacts/frozen_specialist_truncation_ablation_20260810_v1'
TARGET='grade_atr_spacing_2p0'; LEVELS=(5,10,20)
PARAMS={"n_estimators":180,"learning_rate":.03,"max_depth":4,"num_leaves":16,"min_child_samples":776,"min_sum_hessian_in_leaf":28.081,"min_gain_to_split":.00333,"colsample_bytree":.840,"subsample":.730,"subsample_freq":1,"reg_alpha":.000123,"reg_lambda":1.746,"max_bin":127,"label_gain":[0.,.1,1.,3.,7.,12.],"verbosity":-1,"random_state":SEED,"n_jobs":1}
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True); frame,views=_load(Path('data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1/frozen_view_contract.json'),Path('data_perp/artifacts/query_screen_population_20260810_v1.parquet')); tr,va=_development_splits(frame); rows=[]
 for level in LEVELS:
  chunks=[]
  for side in ('long','short'):
   a,b=tr[tr.side_name.eq(side)].copy(),va[va.side_name.eq(side)].copy(); fields=sorted({f for v in views[side].values() for f in v}); a=a.merge(_store_rows(a,fields),on='candidate_id',validate='one_to_one'); b=b.merge(_store_rows(b,fields),on='candidate_id',validate='one_to_one'); q=_query_id(a); x=a[['candidate_id',*fields,TARGET]].copy(); x['q']=q.to_numpy(); x['row']=np.arange(len(x)); x=x.sort_values(['q','candidate_id'],kind='stable'); sizes=x.groupby('q',sort=False).size(); x=x[x.q.isin(sizes.index[sizes.ge(2)])]; groups=x.groupby('q',sort=False).size().to_numpy(np.int32); med=a[fields].apply(pd.to_numeric,errors='coerce').median(); model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',lambdarank_truncation_level=level,**PARAMS); model.fit(x[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.),a[TARGET].to_numpy(float)[x.row.to_numpy()],group=groups); z=b[['candidate_id','__ts__','side_name','net_bps','gross_bps']].copy(); z['score']=model.predict(b[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.)); chunks.append(z)
  p=pd.concat(chunks,ignore_index=True); rows.append({'truncation_level':level,**global_tail_metrics(p),**monthly_stability(p)})
 pd.DataFrame(rows).to_parquet(out/'truncation_metrics.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'frozen_specialist_truncation_ablation_v1','target':TARGET,'query':'q1_cycle_4h_side','levels':list(LEVELS),'selection':'global top5 net, then monthly stability, then top1 net'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); print(run(a.out))
