#!/usr/bin/env python3
"""Final gated meta comparison for correctness-leaf regime representations.

The comparison is intentionally post-gate: only candidates satisfying all
three-fold support/stability/transport requirements may enter `REGIME_GATED`.
If none passes, that arm is an exact baseline clone.  This documents the
negative result without smuggling fold-specific discovery features into the
meta model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_input_20260803_v1/input.parquet'
AVAIL=ROOT/'data_perp/artifacts/correctness_leaf_regime_input_20260803_v1/feature_availability.parquet'
GATES=ROOT/'data_perp/artifacts/correctness_leaf_regime_oof_20260803_v3/accepted_leaf_regime_candidates.parquet'
OUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_meta_comparison_20260803_v1'

def folds(ts):
 values=pd.Index(ts.drop_duplicates().sort_values());return ts.map({x:min(4,int(5*i/max(len(values),1))) for i,x in enumerate(values)}).astype('int8')

def matrix(train,test,fields):
 med=train[fields].replace([np.inf,-np.inf],np.nan).median().fillna(0.)
 return (train[fields].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy('float32'),test[fields].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy('float32'))

def label(residual):
 return np.where(residual<=-50.,0,np.where(residual>=50.,2,1)).astype('int8')

def economics(frame,score,arm,fold):
 out=[]
 for q in (.01,.05,.10):
  z=frame.assign(_score=score).sort_values(['_score','candidate_id'],ascending=[False,True],kind='stable').head(max(1,int(np.ceil(len(frame)*q))))
  for view,x in [('global',z),('long',z[z.side_name.eq('long')]),('short',z[z.side_name.eq('short')])]:
   if len(x):out.append({'arm':arm,'fold':fold,'view':view,'top_fraction':q,'trades':len(x),'net_bps':float(x.net_bps.mean()),'gross_bps':float(x.gross_bps.mean()),'rank_ic':float(spearmanr(score,frame.net_bps).statistic)})
 return out

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 a=pd.read_parquet(AVAIL);fields=a.loc[a.usable_90pct_nonconstant,'feature'].astype(str).tolist()
 excluded={'candidate_id','candidate_key','__ts__','side_name','era','gross_bps','net_bps','prequential_base_expected_net_bps'}
 fields=[x for x in fields if x not in excluded]
 keep=['candidate_id','__ts__','side_name','gross_bps','net_bps','prequential_base_expected_net_bps',*fields]
 d=pd.read_parquet(INPUT,columns=keep);d.__ts__=pd.to_datetime(d.__ts__,utc=True);d['label_available_ts']=d.__ts__+pd.Timedelta(hours=13)
 d=d[np.isfinite(d.net_bps)&np.isfinite(d.prequential_base_expected_net_bps)].copy()
 # This is the defined top-5% base cohort per decision timestamp, not an
 # all-period percentile and not a post-label selection.
 d['cohort']=d.groupby('__ts__')['prequential_base_expected_net_bps'].rank(method='first',pct=True,ascending=False).le(.05)
 d=d[d.cohort].sort_values(['__ts__','candidate_id']).reset_index(drop=True);d['fold']=folds(d.__ts__)
 accepted=pd.read_parquet(GATES) if GATES.exists() else pd.DataFrame()
 if not accepted.empty: raise RuntimeError('This runner requires materialised gated columns; implement only after a feature advances.')
 metrics=[];pred=[];meta=[]
 for fold in (2,3,4):
  te=d[d.fold.eq(fold)].copy();start=te.__ts__.min();trall=d[d.label_available_ts.lt(start)].copy()
  parts=[]
  for side in ('long','short'):
   tr=trall[trall.side_name.eq(side)].copy();x=te[te.side_name.eq(side)].copy()
   # The early outer fold has no short candidates after the strict global
   # top-5% admission.  It is an absent evaluation population, not a failed
   # class and must not be imputed or fabricated.
   if x.empty: continue
   r=tr.net_bps.to_numpy(float)-tr.prequential_base_expected_net_bps.to_numpy(float);y=label(r)
   if set(y)!={0,1,2}:raise RuntimeError(f'missing class {fold}/{side}')
   xx,xt=matrix(tr,x,fields)
   counts=np.bincount(y,minlength=3).astype(float);w=np.sqrt(len(y)/np.maximum(3*counts[y],1))[y];w=np.clip(w/w.mean(),.5,2.)
   model=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=120,learning_rate=.035,num_leaves=20,min_child_samples=max(80,int(.01*len(tr))),colsample_bytree=.8,reg_lambda=20.,random_state=20260803+fold,n_jobs=1,verbosity=-1).fit(xx,y,sample_weight=w)
   p=np.clip(model.predict_proba(xt),1e-6,1.);p/=p.sum(1,keepdims=True)
   means=np.array([r[y==k].mean() for k in range(3)])
   x['meta_correction_bps']=p@means;x['meta_score_bps']=x.prequential_base_expected_net_bps+x.meta_correction_bps
   x['residual_class']=label(x.net_bps.to_numpy(float)-x.prequential_base_expected_net_bps.to_numpy(float));x['p_low'],x['p_mid'],x['p_high']=p.T;parts.append(x)
   meta.append({'fold':fold,'side_name':side,'train_rows':len(tr),'test_rows':len(x),'feature_count':len(fields),'leaf_feature_count':0,'test_log_loss':float(log_loss(x.residual_class,p,labels=[0,1,2]))})
  z=pd.concat(parts,ignore_index=True);pred.append(z)
  for arm,score in [('BASELINE_ALL_META',z.meta_score_bps.to_numpy(float)),('REGIME_GATED',z.meta_score_bps.to_numpy(float))]:metrics.extend(economics(z,score,arm,fold))
 pd.concat(pred,ignore_index=True).to_parquet(OUT/'oof_predictions.parquet',index=False);pd.DataFrame(metrics).to_parquet(OUT/'metrics.parquet',index=False);pd.DataFrame(meta).to_parquet(OUT/'meta_diagnostics.parquet',index=False)
 manifest={'status':'COMPLETED_NEGATIVE_GATE','base':'R3 TP6/SL4','meta_target':'side-local three-state ordinal net residual: <=-50 / (-50,50) / >=50 bps','cohort':'top 5% base score per timestamp','features':{'all_eligible_causal_meta_features':len(fields),'accepted_leaf_regime_features':0},'comparison':'REGIME_GATED is identical to BASELINE_ALL_META because no leaf regime representation passed the predeclared gates; this is not a positive ablation.'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))

if __name__=='__main__':run()
