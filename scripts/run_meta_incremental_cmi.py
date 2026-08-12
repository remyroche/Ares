#!/usr/bin/env python3
"""Incremental binned-CMI meta feature addition restricted to config meta keys."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.config import PERP_META_PRIMARY_FEATURE_KEYS, RESIDUAL_META_FEATURE_KEYS, MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS, T2_FUNNEL_META_CONTEXT_FEATURE_KEYS
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_residual_query_hpo import _load,_fit_specialists,_make_features,_fit_residual,_utc
from scripts.run_frozen_multiview_specialist_input_ablation import _store_rows
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
OUT=ROOT/'data_perp/artifacts/meta_incremental_cmi_20260810_v1'
BASE_FIELDS=['p_clear','p_adverse','p_weak','base_score','prequential_base_expected_net_bps']
PARAMS={"n_estimators":180,"learning_rate":.03,"max_depth":4,"num_leaves":31,"min_child_samples":893,"min_sum_hessian_in_leaf":1.13,"min_gain_to_split":.0089,"colsample_bytree":.79,"subsample":.87,"subsample_freq":1,"reg_alpha":.03,"reg_lambda":.17,"max_bin":63,"label_gain":[0.,.25,1.,3.,7.,12.],"verbosity":-1,"random_state":20260810,"n_jobs":1}
def _bin(v):
 v=np.asarray(v,float); ok=np.isfinite(v); out=np.full(len(v),-1,np.int8)
 if ok.any():
  q=np.nanquantile(v[ok],[.25,.5,.75]); out[ok]=np.digitize(v[ok],q).astype(np.int8)
 return out
def _cmi(x,y,z):
 x,y,z=_bin(x),_bin(y),_bin(z); ok=(x>=0)&(y>=0)&(z>=0)
 if ok.sum()<100: return -np.inf
 a=np.stack([x[ok],y[ok],z[ok]],axis=1); n=float(len(a)); cxyz={}; cxz={}; cyz={}; cz={}
 for i,j,k in map(tuple,a): cxyz[(i,j,k)]=cxyz.get((i,j,k),0)+1; cxz[(i,k)]=cxz.get((i,k),0)+1; cyz[(j,k)]=cyz.get((j,k),0)+1; cz[k]=cz.get(k,0)+1
 val=0.
 for (i,j,k),c in cxyz.items(): p=c/n; val+=p*math.log(max(c*cz[k]/max(cxz[(i,k)]*cyz[(j,k)],1e-12),1e-12))
 return float(val)
def _candidates(store_cols):
 keys=list(dict.fromkeys(PERP_META_PRIMARY_FEATURE_KEYS+RESIDUAL_META_FEATURE_KEYS+MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS+T2_FUNNEL_META_CONTEXT_FEATURE_KEYS)); return [k for k in keys if k in store_cols]
def _store_schema():
 import duckdb
 from scripts.run_frozen_multiview_specialist_input_ablation import STORE
 con=duckdb.connect(); cols={str(r[0]) for r in con.execute('describe select * from read_parquet(?)',[str(STORE)]).fetchall()}; con.close(); return cols
def run(out=OUT,max_steps=8):
 out.mkdir(parents=True,exist_ok=True); base,views,ae,ctx=_load(); candidates=_candidates(_store_schema()); all_pred=[]; selected_rows=[]
 for fold in LONG_HISTORY_FOLDS[3:]:
  a,b,c,e=map(_utc,(fold.train_start,fold.calibration_start,fold.test_start,fold.test_end)); tr=base[base.__ts__.between(a,b,inclusive='left')&base.label_available_ts.lt(b)]; ca=base[base.__ts__.between(b,c,inclusive='left')&base.label_available_ts.lt(c)]; te=base[base.__ts__.between(c,e,inclusive='left')]
  for side in ('long','short'):
   train,cal,test=(x[x.side_name.eq(side)].copy() for x in (tr,ca,te)); cs,ts=_fit_specialists(train,cal,test,views[side],'q4h_side'); cal0,fields0=_make_features(cal,cs,[]); test0,_=_make_features(test,ts,[]); cal_feat=_store_rows(cal,candidates); test_feat=_store_rows(test,candidates); cal0=cal0.merge(cal_feat,on='candidate_id',validate='one_to_one'); test0=test0.merge(test_feat,on='candidate_id',validate='one_to_one'); split=max(1,len(cal0)//2); select=cal0.iloc[:split].copy(); fit=cal0.iloc[split:].copy(); residual=select.net_bps.to_numpy(float)-select.prequential_base_expected_net_bps.to_numpy(float); z=select.base_score.to_numpy(float); ranked=[]; remaining=[]
   for f in candidates:
    v=pd.to_numeric(select[f],errors='coerce'); cov=float(v.notna().mean()); scale=float((v-v.median()).abs().median()) if v.notna().any() else 0.;
    if cov>=.90 and np.isfinite(scale) and scale>1e-8: remaining.append((f,_cmi(v.to_numpy(float),residual,z)))
   for step in range(max_steps):
    if not remaining: break
    remaining.sort(key=lambda t:(-t[1],t[0])); chosen,score=remaining.pop(0); ranked.append(chosen); selected_rows.append({'fold':fold.name,'side':side,'step':step+1,'feature':chosen,'cmi':score})
    flds=BASE_FIELDS+[c for c in fields0 if c.startswith('mv__')]+ranked; flds=[f for f in flds if f in fit.columns and f in test0.columns]; raw=_fit_residual(fit,test0,flds,'q4h_side',PARAMS); p=test[['candidate_id','__ts__','side_name','net_bps','gross_bps','prequential_base_expected_net_bps']].copy(); p['score']=test.prequential_base_expected_net_bps.to_numpy(float)+raw; p['fold']=fold.name; p['side_model']=side; p['step']=step+1; p['added_feature']=chosen; all_pred.append(p)
   del cal_feat,test_feat,cal0,test0; import gc; gc.collect()
 pred=pd.concat(all_pred,ignore_index=True); pred.to_parquet(out/'predictions.parquet',index=False); pd.DataFrame(selected_rows).to_parquet(out/'selected_features.parquet',index=False); rows=[]
 for step,g in pred.groupby('step',sort=True): rows.append({'step':step,'added_feature':g.added_feature.iloc[0],**global_tail_metrics(g),**monthly_stability(g)})
 pd.DataFrame(rows).to_parquet(out/'metrics.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'meta_incremental_cmi_v1','candidate_keys':'config.PERP_META_PRIMARY_FEATURE_KEYS + RESIDUAL_META_FEATURE_KEYS + MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS + T2_FUNNEL_META_CONTEXT_FEATURE_KEYS','conditional_mi':'binned I(feature; residual | base_score) proxy; selection half only','max_depth':4,'steps':max_steps,'specialist_target':'grade_atr_spacing_2p0','query':'q4h_side'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); ap.add_argument('--max-steps',type=int,default=8); a=ap.parse_args(); print(run(a.out,a.max_steps))
