#!/usr/bin/env python3
"""Larger-feature regime-grouped specialist heads through residual meta."""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.config import SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_multiview_specialist_input_ablation import STORE, _store_rows, _base, _utc
from scripts.run_frozen_residual_query_hpo import _fit_residual
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
OUT=ROOT/'data_perp/artifacts/regime_grouped_larger_specialists_20260810_v1'
PARAMS={"n_estimators":180,"learning_rate":.03,"max_depth":4,"num_leaves":24,"min_child_samples":600,"colsample_bytree":.8,"subsample":.8,"subsample_freq":1,"reg_lambda":15.,"verbosity":-1,"random_state":20260810,"n_jobs":1}
RESIDUAL_PARAMS={"n_estimators":220,"learning_rate":.03,"max_depth":4,"num_leaves":31,"min_child_samples":893,"min_sum_hessian_in_leaf":1.13,"min_gain_to_split":.0089,"colsample_bytree":.79,"subsample":.87,"subsample_freq":1,"reg_alpha":.03,"reg_lambda":.17,"max_bin":63,"label_gain":[0.,.25,1.,3.,7.,12.],"verbosity":-1,"random_state":20260810,"n_jobs":1}
BASE_FIELDS=['p_clear','p_adverse','p_weak','base_score','prequential_base_expected_net_bps']
REGIMES=('volatility','trend','transition','entropy','composite')
def _schema():
 import duckdb
 con=duckdb.connect(); cols={str(r[0]) for r in con.execute('describe select * from read_parquet(?)',[str(STORE)]).fetchall()}; con.close(); return cols
def _large_fields(train, cols):
 candidates=[x for x in dict.fromkeys(SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS) if x in cols]
 sample=train.sample(min(20000,len(train)),random_state=20260810)
 mat=_store_rows(sample,candidates); scores=[]
 for c in candidates:
  v=pd.to_numeric(mat[c],errors='coerce'); cov=float(v.notna().mean()); med=float(v.median()) if v.notna().any() else 0.; scale=float((v-med).abs().median()) if v.notna().any() else 0.;
  if cov>=.90 and np.isfinite(scale) and scale>1e-8: scores.append((scale,c))
 scores.sort(reverse=True); return [c for _,c in scores[:160]]
def _regime_columns(mat):
 tr=[c for c in mat.columns if c.startswith('mkt_regime_change__')]
 out=pd.DataFrame(index=mat.index)
 for c in tr:
  v=pd.to_numeric(mat[c],errors='coerce'); out[c]=v
 if tr:
  arr=out[tr].to_numpy(float); finite=np.isfinite(arr); signs=(arr>=0).astype(float); p=np.nanmean(np.where(finite,signs,np.nan),axis=1); p=np.clip(np.nan_to_num(p,nan=.5),1e-4,1-1e-4); out['transition_intensity']=np.nan_to_num(np.nanmean(np.where(finite,np.abs(arr),np.nan),axis=1),nan=0.); out['transition_entropy']=-(p*np.log(p)+(1-p)*np.log(1-p));
 else: out['transition_intensity']=0.; out['transition_entropy']=0.
 out['volatility_proxy']=pd.to_numeric(mat.get('atr_percentile',pd.Series(0.,index=mat.index)),errors='coerce').fillna(0.)
 out['trend_proxy']=pd.to_numeric(mat.get('trend_strength_percentile',mat.get('trend_slope_48h',pd.Series(0.,index=mat.index))),errors='coerce').fillna(0.)
 return out[['volatility_proxy','trend_proxy','transition_intensity','transition_entropy']]
def _bins(train_vals, vals):
 q=np.nanquantile(train_vals[np.isfinite(train_vals)],[.25,.5,.75]) if np.isfinite(train_vals).any() else np.array([0.,0.,0.]); return np.digitize(np.nan_to_num(vals,nan=float(np.nanmedian(train_vals)) if np.isfinite(train_vals).any() else 0.),q)
def _rank_fit(train, fields, target, q):
 x=train[['candidate_id',*fields]].copy(); x['q']=q.to_numpy(); x['row']=np.arange(len(x)); x=x.sort_values(['q','candidate_id'],kind='stable'); sizes=x.groupby('q',sort=False).size(); x=x[x.q.isin(sizes.index[sizes.ge(2)])]; groups=x.groupby('q',sort=False).size().to_numpy(np.int32); med=train[fields].apply(pd.to_numeric,errors='coerce').median(); model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',lambdarank_truncation_level=10,**PARAMS); model.fit(x[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.),target[x.row.to_numpy()],group=groups); return model,med
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True); base=_base(); cols=_schema(); all_pred=[]; audit=[]
 # One larger specialist feature contract is frozen from the earliest transport
 # training population and reused in every fold.
 first=LONG_HISTORY_FOLDS[3]; start,end=_utc(first.train_start),_utc(first.calibration_start); template=base[base.__ts__.between(start,end,inclusive='left')]; fields=_large_fields(template,cols); transition=[c for c in cols if c.startswith('mkt_regime_change__')]; regime_store=['atr_percentile','trend_strength_percentile','trend_slope_48h',*transition]; regime_store=[c for c in regime_store if c in cols]
 (out/'feature_contract.json').write_text(json.dumps({'feature_count':len(fields),'features':fields,'regime_fields':regime_store,'query_modes':list(REGIMES)},indent=2)+'\n')
 for fold in LONG_HISTORY_FOLDS[3:]:
  a,b,c,e=map(_utc,(fold.train_start,fold.calibration_start,fold.test_start,fold.test_end)); tr=base[base.__ts__.between(a,b,inclusive='left')&base.label_available_ts.lt(b)]; ca=base[base.__ts__.between(b,c,inclusive='left')&base.label_available_ts.lt(c)]; te=base[base.__ts__.between(c,e,inclusive='left')]
  for side in ('long','short'):
   train,cal,test=(x[x.side_name.eq(side)].copy() for x in (tr,ca,te)); train=train.sample(min(150000,len(train)),random_state=20260810); fs=train.merge(_store_rows(train,fields+regime_store),on='candidate_id',validate='one_to_one'); cs=cal.merge(_store_rows(cal,fields+regime_store),on='candidate_id',validate='one_to_one'); ts=test.merge(_store_rows(test,fields+regime_store),on='candidate_id',validate='one_to_one'); fr,cr,trg=_regime_columns(fs[regime_store]),_regime_columns(cs[regime_store]),_regime_columns(ts[regime_store]);
   for key in REGIMES:
    if key=='volatility': basekey='volatility_proxy'
    elif key=='trend': basekey='trend_proxy'
    elif key=='transition': basekey='transition_intensity'
    elif key=='entropy': basekey='transition_entropy'
    else: basekey=None
    symbol_train=fs.candidate_id.astype(str).str.split('|').str[0]; symbol_cal=cs.candidate_id.astype(str).str.split('|').str[0]; symbol_test=ts.candidate_id.astype(str).str.split('|').str[0]
    if basekey is None: qtr=(symbol_train+'|'+fr.volatility_proxy.round().astype(str)+'|'+fr.trend_proxy.round().astype(str)+'|'+fr.transition_intensity.round().astype(str)+'|'+fr.transition_entropy.round().astype(str)); qcal=(symbol_cal+'|'+cr.volatility_proxy.round().astype(str)+'|'+cr.trend_proxy.round().astype(str)+'|'+cr.transition_intensity.round().astype(str)+'|'+cr.transition_entropy.round().astype(str)); qtest=(symbol_test+'|'+trg.volatility_proxy.round().astype(str)+'|'+trg.trend_proxy.round().astype(str)+'|'+trg.transition_intensity.round().astype(str)+'|'+trg.transition_entropy.round().astype(str))
    else:
     bt=_bins(fr[basekey].to_numpy(float),fr[basekey].to_numpy(float)); bc=_bins(fr[basekey].to_numpy(float),cr[basekey].to_numpy(float)); be=_bins(fr[basekey].to_numpy(float),trg[basekey].to_numpy(float)); qtr= symbol_train+'|'+pd.Series(bt,index=fs.index).astype(str); qcal=symbol_cal+'|'+pd.Series(bc,index=cs.index).astype(str); qtest=symbol_test+'|'+pd.Series(be,index=ts.index).astype(str)
    target=(fs.net_bps.to_numpy(float)>50.).astype(np.int32); model,med=_rank_fit(fs,fields,target,qtr); sc=model.predict(cs[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.)); st=model.predict(ts[fields].apply(pd.to_numeric,errors='coerce').fillna(med).fillna(0.));
    z=test[['candidate_id','__ts__','side_name','net_bps','gross_bps','prequential_base_expected_net_bps']].copy(); z['score']=st; z['regime_query']=key; z['fold']=fold.name; z['level']='standalone'; all_pred.append(z)
    cal_meta=cal[['candidate_id','__ts__','side_name','net_bps','gross_bps',*BASE_FIELDS]].copy(); test_meta=test[['candidate_id','__ts__','side_name','net_bps','gross_bps',*BASE_FIELDS]].copy(); cal_meta['regime_specialist_score']=sc; test_meta['regime_specialist_score']=st; residual_score=_fit_residual(cal_meta,test_meta,BASE_FIELDS+['regime_specialist_score'],'q4h_side',RESIDUAL_PARAMS); rz=test[['candidate_id','__ts__','side_name','net_bps','gross_bps','prequential_base_expected_net_bps']].copy(); rz['score']=test.prequential_base_expected_net_bps.to_numpy(float)+residual_score; rz['regime_query']=key; rz['fold']=fold.name; rz['level']='residual'; all_pred.append(rz); audit.append({'fold':fold.name,'side':side,'query':key,'feature_count':len(fields),'train_rows':len(fs)})
   del fs,cs,ts; gc.collect()
 p=pd.concat(all_pred,ignore_index=True); p.to_parquet(out/'predictions.parquet',index=False); pd.DataFrame(audit).to_parquet(out/'audit.parquet',index=False); rows=[]
 for (q,level),g in p.groupby(['regime_query','level'],sort=False): rows.append({'query':q,'level':level,**global_tail_metrics(g),**monthly_stability(g)})
 pd.DataFrame(rows).to_parquet(out/'metrics.parquet',index=False); (out/'manifest.json').write_text(json.dumps({'schema':'regime_grouped_larger_specialists_v1','target':'binary_h12_net50','feature_count':len(fields),'query_modes':list(REGIMES),'note':'specialist standalone scores are materialized; downstream residual replay is a follow-up arm using the same frozen scores'},indent=2)+'\n'); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); print(run(a.out))
