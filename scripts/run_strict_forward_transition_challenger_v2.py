#!/usr/bin/env python3
"""Train-only blocked-CV challenger to strict transition forward baseline v1."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, uuid
from pathlib import Path
from typing import Any, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_strict_forward_transition_evaluation import ART, CATALOGUE, CURRENT, TRAIN_END, causal_feature_columns, ece, global_top10, label_available, safe, sha256

OUT=ART/'strict_forward_transition_challenger_20260730_v2'; BASELINE=ART/'strict_forward_transition_evaluation_20260730_v1/forward_transition_predictions.parquet'
FOLDS=(pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2024-07-01',tz='UTC'),pd.Timestamp('2025-01-01',tz='UTC'),pd.Timestamp('2025-07-01',tz='UTC'))
FAMILY=('structure','dynamics','all_causal')

def family_features(frame:pd.DataFrame, train:pd.DataFrame, family:str)->list[str]:
 base=causal_feature_columns(frame,train)
 if family=='structure': keep=[x for x in base if not x.startswith(('mkt_regime_change__','transition_new__'))]
 elif family=='dynamics': keep=[x for x in base if x.startswith(('mkt_regime_change__','transition_new__','market_state_','breakout_','compression_','recovery_'))]
 else: keep=base
 return keep[:32] if keep else base[:16]

def model(kind:str, *, multiclass:bool, positive_weight:float=1., seed:int=0):
 if kind=='lgbm':
  return lgb.LGBMClassifier(n_estimators=80,learning_rate=.06,num_leaves=15,min_child_samples=40,reg_lambda=2.,subsample=.9,colsample_bytree=.85,n_jobs=1,random_state=seed,verbosity=-1,objective='multiclass' if multiclass else 'binary')
 return HistGradientBoostingClassifier(max_iter=48,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.,random_state=seed)

def binary_fold_metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
 return {'ap':float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan,'brier':float(brier_score_loss(y,p)),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,'ece10':ece(pd.Series(y),pd.Series(p))}

def platt(train:pd.DataFrame, test:pd.DataFrame)->np.ndarray:
 if len(train)<20 or train.y.nunique()<2:return test.raw.to_numpy(float)
 c=LogisticRegression(C=1.,max_iter=200,random_state=20260730).fit(train[['raw']],train.y)
 return c.predict_proba(test[['raw']])[:,1]

def active_trials(frame:pd.DataFrame)->tuple[pd.DataFrame,dict[tuple[str,str,float],pd.DataFrame]]:
 records=[]; predictions={}
 for family in FAMILY:
  for kind in ('hgb','lgbm'):
   for weight in (1.,5.):
    trial=[]
    for number,start in enumerate(FOLDS):
     stop=start+pd.DateOffset(months=6); train=frame.loc[frame.source_utc.lt(start)].copy(); test=frame.loc[frame.source_utc.ge(start)&frame.source_utc.lt(stop)].copy(); features=family_features(frame,train,family)
     imp=SimpleImputer(strategy='median'); x=imp.fit_transform(train[features]); z=imp.transform(test[features]); y=train.target__transition_active.astype(int).to_numpy(); w=np.where(y==1,weight,1.)
     fitted=model(kind,multiclass=False,positive_weight=weight,seed=20260730+number).fit(x,y,sample_weight=w); raw=fitted.predict_proba(z)[:,list(fitted.classes_).index(1)]
     trial.append(pd.DataFrame({'fold':number,'era':start.year,'y':test.target__transition_active.astype(int).to_numpy(),'raw':raw}))
    raw=pd.concat(trial,ignore_index=True); calibrated=[]
    for number,g in raw.groupby('fold',sort=True):
     prior=raw.loc[raw.fold.lt(number)]; x=g.copy();x['probability']=platt(prior,x);calibrated.append(x)
    pred=pd.concat(calibrated,ignore_index=True); per=[]
    for fold,g in pred.groupby('fold',sort=True): per.append({'fold':fold,**binary_fold_metrics(g.y.to_numpy(),g.probability.to_numpy())})
    per=pd.DataFrame(per); composite=per.ap-per.brier
    records.append({'family':family,'model':kind,'positive_weight':weight,'mean_ap':per.ap.mean(),'mean_brier':per.brier.mean(),'mean_ece10':per.ece10.mean(),'mean_composite':composite.mean(),'min_fold_composite':composite.min(),'folds_with_positive':int(per.ap.notna().sum())}); predictions[(family,kind,weight)]=pred
 return pd.DataFrame(records),predictions

def lifecycle_trials(frame:pd.DataFrame)->pd.DataFrame:
 records=[]
 for family in FAMILY:
  for kind in ('hgb','lgbm'):
   for balanced in (False,True):
    rows=[]
    for number,start in enumerate(FOLDS):
     stop=start+pd.DateOffset(months=6); train=frame.loc[frame.source_utc.lt(start)&frame.target__pattern_phase.notna()].copy(); test=frame.loc[frame.source_utc.ge(start)&frame.source_utc.lt(stop)&frame.target__pattern_phase.notna()].copy(); features=family_features(frame,train,family)
     imp=SimpleImputer(strategy='median');x=imp.fit_transform(train[features]);z=imp.transform(test[features]);y=train.target__pattern_phase.astype(str); w=np.ones(len(y))
     if balanced:
      counts=y.value_counts();w=y.map(lambda c:len(y)/(len(counts)*counts[c])).to_numpy(float)
     fitted=model(kind,multiclass=True,seed=20260800+number).fit(x,y,sample_weight=w);p=fitted.predict_proba(z);labels=fitted.classes_.astype(str);pred=labels[np.argmax(p,axis=1)]; actual=test.target__pattern_phase.astype(str).to_numpy()
     rows.append({'fold':number,'macro_f1':f1_score(actual,pred,average='macro',zero_division=0),'accuracy':accuracy_score(actual,pred),'log_loss':log_loss(actual,p,labels=labels)})
    m=pd.DataFrame(rows);records.append({'family':family,'model':kind,'balanced':balanced,'mean_macro_f1':m.macro_f1.mean(),'min_fold_macro_f1':m.macro_f1.min(),'mean_accuracy':m.accuracy.mean(),'mean_log_loss':m.log_loss.mean()})
 return pd.DataFrame(records)

def run(*,catalogue:Path=CATALOGUE,current:Path=CURRENT,baseline:Path=BASELINE,output:Path=OUT)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 f=pd.read_parquet(catalogue).copy();f.source_utc=pd.to_datetime(f.source_utc,utc=True); latest=pd.to_datetime(pd.read_parquet(current,columns=['__ts__'])['__ts__'].max(),utc=True);resolved=label_available(f);train=f.loc[f.source_utc.lt(TRAIN_END)&resolved.lt(TRAIN_END)&f.target__transition_active.notna()].copy();test=f.loc[f.source_utc.ge(TRAIN_END)&f.source_utc.le(latest)&f.target__transition_active.notna()].copy()
 active_hpo,active_oof=active_trials(train); winner=active_hpo.sort_values(['mean_composite','min_fold_composite','mean_ap'],ascending=False,kind='stable').iloc[0]; key=(winner.family,winner.model,float(winner.positive_weight));calibration=active_oof[key]; calibrator=LogisticRegression(C=1.,max_iter=200,random_state=20260730).fit(calibration[['raw']],calibration.y) if calibration.y.nunique()==2 else None
 lifecycle_hpo=lifecycle_trials(train);life_winner=lifecycle_hpo.sort_values(['mean_macro_f1','min_fold_macro_f1'],ascending=False,kind='stable').iloc[0]
 features=family_features(train,train,winner.family);imp=SimpleImputer(strategy='median');x=imp.fit_transform(train[features]);z=imp.transform(test[features]);y=train.target__transition_active.astype(int).to_numpy();weights=np.where(y==1,float(winner.positive_weight),1.);active=model(winner.model,multiclass=False,seed=20260900).fit(x,y,sample_weight=weights);raw=active.predict_proba(z)[:,list(active.classes_).index(1)];prob=calibrator.predict_proba(pd.DataFrame({'raw':raw}))[:,1] if calibrator is not None else raw
 phase_train=train.loc[train.target__pattern_phase.notna()].copy();px=imp.transform(phase_train[features]);py=phase_train.target__pattern_phase.astype(str);pw=np.ones(len(py))
 if bool(life_winner.balanced):
  counts=py.value_counts();pw=py.map(lambda c:len(py)/(len(counts)*counts[c])).to_numpy(float)
 life=model(life_winner.model,multiclass=True,seed=20260901).fit(px,py,sample_weight=pw);lp=life.predict_proba(z);classes=life.classes_.astype(str);out=test.loc[:,['source_utc','target__transition_active','target__pattern_phase']].copy();out['transition_probability']=prob;out['lifecycle_predicted_phase']=classes[np.argmax(lp,axis=1)];out['lifecycle_probability']=lp.max(axis=1)
 metrics=[]
 for scope,g in [('all_2026',out),*[(f'month::{m}',x) for m,x in out.assign(month=out.source_utc.dt.strftime('%Y-%m')).groupby('month',sort=True)]]:metrics.append({'arm':'v2', 'scope':scope,**binary_fold_metrics(g.target__transition_active.astype(int).to_numpy(),g.transition_probability.to_numpy())})
 base=pd.read_parquet(baseline,columns=['source_utc','target__transition_active','target__pattern_phase','transition_probability','lifecycle_predicted_phase']);base.source_utc=pd.to_datetime(base.source_utc,utc=True)
 for scope,g in [('all_2026',base),*[(f'month::{m}',x) for m,x in base.assign(month=base.source_utc.dt.strftime('%Y-%m')).groupby('month',sort=True)]]:metrics.append({'arm':'v1', 'scope':scope,**binary_fold_metrics(g.target__transition_active.astype(int).to_numpy(),g.transition_probability.to_numpy())})
 life_eval=out.loc[out.target__pattern_phase.notna()];base_eval=base.loc[base.target__pattern_phase.notna()];life_metrics=pd.DataFrame([{'arm':'v2','rows':len(life_eval),'accuracy':accuracy_score(life_eval.target__pattern_phase.astype(str),life_eval.lifecycle_predicted_phase),'macro_f1':f1_score(life_eval.target__pattern_phase.astype(str),life_eval.lifecycle_predicted_phase,average='macro',zero_division=0)},{'arm':'v1','rows':len(base_eval),'accuracy':accuracy_score(base_eval.target__pattern_phase.astype(str),base_eval.lifecycle_predicted_phase),'macro_f1':f1_score(base_eval.target__pattern_phase.astype(str),base_eval.lifecycle_predicted_phase,average='macro',zero_division=0)}])
 c=pd.read_parquet(current,columns=['candidate_id','__ts__','execution_net_ev_12h','catboost__residual__without_hpo__all_features']);c.__ts__=pd.to_datetime(c.__ts__,utc=True);c=c.loc[c.__ts__.le(out.source_utc.max())].copy();c['month']=c.__ts__.dt.strftime('%Y-%m');c['selected_global_top10']=False
 for _,g in c.groupby('month',sort=True):c.loc[g.index,'selected_global_top10']=global_top10(g,'catboost__residual__without_hpo__all_features')
 economy=[]
 for arm,pred in [('v1',base),('v2',out)]:
  e=c.loc[c.selected_global_top10].merge(pred[['source_utc','transition_probability']],left_on='__ts__',right_on='source_utc',how='inner');e['risk_decile']=pd.qcut(e.transition_probability.rank(method='first'),10,labels=False,duplicates='drop');x=e.groupby(['month','risk_decile'],as_index=False).agg(selected_rows=('candidate_id','size'),mean_net_bps=('execution_net_ev_12h',lambda v:float(v.mean()*1e4)),mean_transition_probability=('transition_probability','mean'));x['arm']=arm;economy.append(x)
 stage=output.parent/f'.{output.name}.{uuid.uuid4().hex}.stage';stage.mkdir(parents=True)
 try:
  active_hpo.to_csv(stage/'active_inner_cv_hpo.csv',index=False);lifecycle_hpo.to_csv(stage/'lifecycle_inner_cv_hpo.csv',index=False);pd.DataFrame(metrics).to_csv(stage/'v1_v2_discrimination_calibration.csv',index=False);life_metrics.to_csv(stage/'v1_v2_lifecycle.csv',index=False);pd.concat(economy).to_csv(stage/'v1_v2_global_top10_economics.csv',index=False);out.to_parquet(stage/'v2_forward_predictions.parquet',index=False);(stage/'selected_features.json').write_text(json.dumps(features,indent=2)+'\n')
  manifest={'schema':'strict_forward_transition_challenger_v2','research_only':True,'promotion_eligible':False,'selection_contract':'all family/model/HPO/imbalance/calibration selection is blocked inner-CV on resolved 2022-2025 only; 2026 is one untouched comparison','feature_contract':'causal numeric fields only; current-regime state and all targets/ex-post phases excluded','calibration_contract':'Platt calibrator is fitted only on preceding blocked-fold predictions during selection and on 2022-25 OOF predictions for final freeze','objective':'maximize mean(AP-Brier), tie-break min blocked-fold composite then AP; lifecycle maximize mean then min macro-F1','inputs_sha256':{'catalogue':sha256(catalogue),'current':sha256(current),'v1_predictions':sha256(baseline)},'outputs_sha256':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()},'winner':{'active':winner.to_dict(),'lifecycle':life_winner.to_dict()},'counts':{'train':len(train),'test':len(test),'features':len(features)}}
  (stage/'manifest.json').write_text(json.dumps(safe(manifest),indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(f"{sha256(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return manifest
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--catalogue',type=Path,default=CATALOGUE);p.add_argument('--current',type=Path,default=CURRENT);p.add_argument('--baseline',type=Path,default=BASELINE);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args(argv);print(json.dumps(safe(run(catalogue=a.catalogue,current=a.current,baseline=a.baseline,output=a.output)),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
