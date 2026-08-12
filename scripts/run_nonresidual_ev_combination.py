#!/usr/bin/env python3
"""Non-residual meta target and side-asymmetric EV-mapped score combinations."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_residual_query_hpo import _load,_fit_specialists,_make_features,_utc
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
OUT=ROOT/'data_perp/artifacts/nonresidual_ev_combination_20260810_v1'
BASE_FIELDS=['p_clear','p_adverse','p_weak','base_score','prequential_base_expected_net_bps']
PARAMS={"n_estimators":180,"learning_rate":.03,"max_depth":4,"num_leaves":31,"min_child_samples":893,"min_sum_hessian_in_leaf":1.13,"min_gain_to_split":.0089,"colsample_bytree":.79,"subsample":.87,"subsample_freq":1,"reg_alpha":.03,"reg_lambda":.17,"max_bin":63,"label_gain":[0.,.1,.25,1.,3.,7.,12.,20.],"verbosity":-1,"random_state":20260810,"n_jobs":1}
WEIGHTS=np.asarray([0.,.25,.5,.75,1.,1.5],float)
NET_EDGES=np.asarray([-200., -100., -50., 0., 50., 100., 200.], float)

def _net_grade(values):
    return np.digitize(np.asarray(values, dtype=float), NET_EDGES, right=True).astype(np.int32)
def _rank_fit(frame,fields,target):
 x=frame[['candidate_id',*fields]].copy(); x['q']=frame.__ts__.dt.floor('4h').astype('int64').astype(str)+'|'+frame.side_name.astype(str); x['row']=np.arange(len(x)); x=x.sort_values(['q','candidate_id'],kind='stable'); sizes=x.groupby('q',sort=False).size(); x=x[x.q.isin(sizes.index[sizes.ge(2)])]; groups=x.groupby('q',sort=False).size().to_numpy(np.int32); med=frame[fields].apply(pd.to_numeric,errors='coerce').median(); model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',lambdarank_truncation_level=10,**PARAMS); model.fit(x[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.),target[x.row.to_numpy()],group=groups); return model,med
def _map(train_score,train_net,apply_score):
 ok=np.isfinite(train_score)&np.isfinite(train_net); s=train_score[ok]; y=train_net[ok]
 if len(s)<100: return np.full(len(apply_score),float(np.nanmean(y) if len(y) else 0.))
 edges=np.unique(np.quantile(s,np.linspace(0,1,11))); bins=np.clip(np.digitize(s,edges[1:-1],right=True),0,9); means=np.array([y[bins==i].mean() if (bins==i).any() else y.mean() for i in range(10)]); return means[np.clip(np.digitize(np.nan_to_num(apply_score,nan=np.nanmedian(s)),edges[1:-1],right=True),0,9)]
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True); base,views,ae,ctx=_load(); preds=[]
 for fold in LONG_HISTORY_FOLDS[3:]:
  a,b,c,e=map(_utc,(fold.train_start,fold.calibration_start,fold.test_start,fold.test_end)); tr=base[base.__ts__.between(a,b,inclusive='left')&base.label_available_ts.lt(b)]; ca=base[base.__ts__.between(b,c,inclusive='left')&base.label_available_ts.lt(c)]; te=base[base.__ts__.between(c,e,inclusive='left')]
  for side in ('long','short'):
   train,cal,test=(x[x.side_name.eq(side)].copy() for x in (tr,ca,te)); cs,ts=_fit_specialists(train,cal,test,views[side],'q4h_side'); cx,fields=_make_features(cal,cs,[]); tx,_=_make_features(test,ts,[]); split=max(1,len(cx)//2); fit=cx.iloc[:split].copy(); mapping=cx.iloc[split:].copy(); target=_net_grade(fit.net_bps.to_numpy(float)); fields=[f for f in BASE_FIELDS+[c for c in fields if c.startswith('mv__')] if f in fit.columns]; model,med=_rank_fit(fit,fields,target); raw_map=model.predict(mapping[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.)); raw_test=model.predict(tx[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.)); meta_map=_map(raw_map,mapping.net_bps.to_numpy(float),raw_test); base_map=_map(mapping.prequential_base_expected_net_bps.to_numpy(float),mapping.net_bps.to_numpy(float),tx.prequential_base_expected_net_bps.to_numpy(float)); z=test[['candidate_id','__ts__','side_name','net_bps','gross_bps']].copy(); z['base_ev']=base_map; z['meta_ev']=meta_map; z['fold']=fold.name; preds.append(z)
 p=pd.concat(preds,ignore_index=True); rows=[]
 for wb in WEIGHTS:
  for wm in WEIGHTS:
   z=p.copy(); z['score']=wb*z.base_ev+wm*z.meta_ev; rows.append({'base_weight':wb,'meta_weight':wm,**global_tail_metrics(z),**monthly_stability(z)})
 table=pd.DataFrame(rows).sort_values(['top5_net_bps','month_std_net_bps','top1_net_bps'],ascending=[False,True,False]); table.to_parquet(out/'combination_metrics.parquet',index=False); p.to_parquet(out/'mapped_predictions.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'nonresidual_ev_combination_v1','meta_target':'non-residual exact H12 net ordinal LambdaRank','query':'q4h_side','ev_mapping':'side-local calibration-half quantile map to net bps','base_weights':WEIGHTS.tolist(),'meta_weights':WEIGHTS.tolist(),'max_depth':4,'selection':'global top5 net, then monthly stability, then top1 net'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); print(run(a.out))
