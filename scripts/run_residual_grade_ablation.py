#!/usr/bin/env python3
"""Residual relevance-grade ablation on the frozen ATR2 specialist stack."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_residual_query_hpo import _load, _fit_specialists, _make_features, _fit_residual
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
OUT=ROOT/'data_perp/artifacts/frozen_residual_grade_ablation_20260810_v1'
PARAMS={"n_estimators":220,"learning_rate":.03,"max_depth":5,"num_leaves":52,"min_child_samples":893,"min_sum_hessian_in_leaf":1.1298052513600887,"min_gain_to_split":.0089300561896448,"colsample_bytree":.7882182037573211,"subsample":.8666554346312396,"subsample_freq":1,"reg_alpha":.030925476912139326,"reg_lambda":.16986488135579808,"max_bin":63,"label_gain":[0.,.25,1.,3.,7.,12.],"verbosity":-1,"random_state":20260810,"n_jobs":1}
GRADES={'default_ordinal':(-150.,-50.,50.,150.),'tight_economic':(-100.,-50.,50.,100.),'wide_economic':(-200.,-75.,75.,200.),'symmetric_50':(-50.,-25.,25.,50.)}
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True); base,views,ae,ctx=_load(); all_rows=[]
 for fold in LONG_HISTORY_FOLDS[3:]:
  from scripts.run_frozen_residual_query_hpo import _utc
  a,b,c,e=map(_utc,(fold.train_start,fold.calibration_start,fold.test_start,fold.test_end)); tr=base[base.__ts__.between(a,b,inclusive='left')&base.label_available_ts.lt(b)]; ca=base[base.__ts__.between(b,c,inclusive='left')&base.label_available_ts.lt(c)]; te=base[base.__ts__.between(c,e,inclusive='left')]
  packs={}
  for side in ('long','short'):
   train,cal,test=(x[x.side_name.eq(side)].copy() for x in (tr,ca,te)); cs,ts=_fit_specialists(train,cal,test,views[side],'q4h_side'); cx,fields=_make_features(cal,cs,ae+ctx); tx,_=_make_features(test,ts,ae+ctx)
   for name,edges in GRADES.items():
    raw=_fit_residual(cx,tx,fields,'q4h_side',PARAMS,grade_edges=edges); z=test[['candidate_id','__ts__','side_name','net_bps','gross_bps']].copy(); z['score']=test.prequential_base_expected_net_bps.to_numpy(float)+raw; z['fold']=fold.name; z['grade_definition']=name; packs.setdefault(name,[]).append(z)
  for name,pieces in packs.items():
   p=pd.concat(pieces,ignore_index=True); all_rows.append(p)
 pred=pd.concat(all_rows,ignore_index=True); pred.to_parquet(out/'predictions.parquet',index=False); rows=[]
 for name,p in pred.groupby('grade_definition',sort=False): rows.append({'grade_definition':name,**global_tail_metrics(p),**monthly_stability(p)})
 pd.DataFrame(rows).to_parquet(out/'metrics.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'frozen_residual_grade_ablation_v1','specialist_target':'grade_atr_spacing_2p0','specialist_query':'q4h_side','residual_query':'q4h_side','grade_definitions':GRADES,'params':PARAMS,'selection':'global top5 net, then monthly stability, then top1 net'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); print(run(a.out))
