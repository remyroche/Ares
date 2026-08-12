#!/usr/bin/env python3
"""Matched downstream impact of specialist query construction."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_residual_query_hpo import _load, _fold_scores
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
OUT=ROOT/'data_perp/artifacts/frozen_specialist_query_residual_impact_20260810_v1'
RESIDUAL_PARAMS={"n_estimators":220,"learning_rate":.03,"max_depth":5,"num_leaves":52,"min_child_samples":893,"min_sum_hessian_in_leaf":1.1298052513600887,"min_gain_to_split":.0089300561896448,"colsample_bytree":.7882182037573211,"subsample":.8666554346312396,"subsample_freq":1,"reg_alpha":.030925476912139326,"reg_lambda":.16986488135579808,"max_bin":63,"label_gain":[0.,.25,1.,3.,7.,12.],"verbosity":-1,"random_state":20260810,"n_jobs":1}
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True); base,views,ae,ctx=_load(); rows=[]
 for query in ('timestamp_side','q4h_side'):
  pieces=[_fold_scores(base,views,ae,ctx,f,query,'q4h_side',RESIDUAL_PARAMS) for f in LONG_HISTORY_FOLDS[3:]]; p=pd.concat(pieces,ignore_index=True); p['specialist_query']=query; p.to_parquet(out/f'predictions_{query}.parquet',index=False); rows.append({'specialist_query':query,**global_tail_metrics(p),**monthly_stability(p)})
 pd.DataFrame(rows).to_parquet(out/'metrics.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'frozen_specialist_query_residual_impact_v1','residual_query':'q4h_side','specialist_queries':['timestamp_side','q4h_side'],'selection':'global top5 net, then monthly stability, then top1 net'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); print(run(a.out))
