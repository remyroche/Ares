#!/usr/bin/env python3
"""Train-only multi-horizon onset and competing-risk transition ablations."""
from __future__ import annotations
import argparse,json,os,sys,shutil,uuid
from pathlib import Path
from typing import Sequence
import lightgbm as lgb
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score,f1_score,accuracy_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_strict_forward_transition_evaluation import ART,CATALOGUE,CURRENT,TRAIN_END,causal_feature_columns,ece,global_top10,label_available,safe,sha256
OUT=ART/'strict_transition_v3_multihorizon_competing_risk_20260730_v2';FOLDS=(pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2024-07-01',tz='UTC'),pd.Timestamp('2025-01-01',tz='UTC'),pd.Timestamp('2025-07-01',tz='UTC'))
HORIZONS=(1,3,6,12)
def fit(x,y,w=None,seed=0,multi=False):return lgb.LGBMClassifier(n_estimators=80,learning_rate=.06,num_leaves=15,min_child_samples=40,reg_lambda=2,n_jobs=1,random_state=seed,verbosity=-1,objective='multiclass' if multi else 'binary').fit(x,y,sample_weight=w)
def metrics(y,p):return {'ap':average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,'auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,'brier':brier_score_loss(y,p),'ece10':ece(pd.Series(y),pd.Series(p))}
def stable_features(frame,train):
 base=causal_feature_columns(frame,train); years=train.source_utc.dt.year
 score=[]
 for c in base:
  means=train.groupby(years)[c].mean();scale=train[c].std()
  if len(means)>=2 and scale>1e-12:score.append((float(means.std()/scale),c))
 return [c for _,c in sorted(score)[:24]]
def platt(prior,raw):
 if len(prior)<20 or prior.y.nunique()<2:return raw
 return LogisticRegression(max_iter=200,random_state=20260730).fit(prior[['raw']],prior.y).predict_proba(pd.DataFrame({'raw':raw}))[:,1]
def cv_horizon(frame,target,features,positive_weight):
 rows=[];preds=[]
 for n,start in enumerate(FOLDS):
  stop=start+pd.DateOffset(months=6);tr=frame[frame.source_utc<start];te=frame[(frame.source_utc>=start)&(frame.source_utc<stop)];imp=SimpleImputer(strategy='median');x=imp.fit_transform(tr[features]);z=imp.transform(te[features]);y=tr[target].astype(int).to_numpy();w=np.where(y==1,positive_weight,1.);m=fit(x,y,w,n);raw=m.predict_proba(z)[:,list(m.classes_).index(1)];preds.append(pd.DataFrame({'fold':n,'y':te[target].astype(int).to_numpy(),'raw':raw}))
 p=pd.concat(preds,ignore_index=True)
 for calibration in ('none','platt'):
  out=[]
  for n,g in p.groupby('fold'):
   q=g.copy();q['p']=q.raw if calibration=='none' else platt(p[p.fold<n],q.raw.to_numpy());out.append(q)
  q=pd.concat(out);per=[]
  for n,g in q.groupby('fold'):per.append(metrics(g.y.to_numpy(),g.p.to_numpy()))
  m=pd.DataFrame(per);rows.append({'target':target,'family':'stable' if len(features)==24 else 'all','positive_weight':positive_weight,'calibration':calibration,'mean_ap':m.ap.mean(),'mean_brier':m.brier.mean(),'mean_composite':(m.ap-m.brier).mean(),'min_fold_composite':(m.ap-m.brier).min()})
 return pd.DataFrame(rows),p
def run(*,catalogue:Path=CATALOGUE,current:Path=CURRENT,output:Path=OUT):
 if output.exists():raise FileExistsError(output)
 f=pd.read_parquet(catalogue).copy();f.source_utc=pd.to_datetime(f.source_utc,utc=True);a=label_available(f);latest=pd.to_datetime(pd.read_parquet(current,columns=['__ts__'])['__ts__'].max(),utc=True);train=f[(f.source_utc<TRAIN_END)&(a<TRAIN_END)].copy();test=f[(f.source_utc>=TRAIN_END)&(a<=latest)].copy()
 for h in HORIZONS:f[f'target_h{h}']=pd.to_numeric(f[f'target__onset_within_{h}h'],errors='coerce').fillna(0).astype(int);train[f'target_h{h}']=f.loc[train.index,f'target_h{h}'];test[f'target_h{h}']=f.loc[test.index,f'target_h{h}']
 allf=causal_feature_columns(f,train)[:32];stable=stable_features(f,train);hpo=[];winners=[]
 for h in HORIZONS:
  for family,features in [('all',allf),('stable',stable)]:
   for positive_weight in (1.,5.):
    x,_=cv_horizon(train,f'target_h{h}',features,positive_weight);x['horizon']=h;hpo.append(x)
  z=pd.concat(hpo[-4:]);w=z.sort_values(['mean_composite','min_fold_composite'],ascending=False).iloc[0];winners.append(w)
 hpo=pd.concat(hpo,ignore_index=True);forward=[]
 for w in winners:
  h=int(w.horizon);features=stable if w.family=='stable' else allf;imp=SimpleImputer(strategy='median');x=imp.fit_transform(train[features]);z=imp.transform(test[features]);y=train[f'target_h{h}'].to_numpy();m=fit(x,y,np.where(y==1,float(w.positive_weight),1.),h);raw=m.predict_proba(z)[:,list(m.classes_).index(1)];# final calibration from all train-only blocked OOF
  _,oof=cv_horizon(train,f'target_h{h}',features,float(w.positive_weight));p=raw
  if w.calibration=='platt':p=LogisticRegression(max_iter=200,random_state=20260730).fit(oof[['raw']],oof.y).predict_proba(pd.DataFrame({'raw':raw}))[:,1]
  forward.append(pd.DataFrame({'source_utc':test.source_utc,'horizon_hours':h,'probability':p,'target':test[f'target_h{h}']}))
 forward=pd.concat(forward);report=[]
 for h,g in forward.groupby('horizon_hours'):
  report.append({'head':f'onset_h{h}','scope':'all_2026',**metrics(g.target.to_numpy(),g.probability.to_numpy())})
  for m,x in g.assign(month=g.source_utc.dt.strftime('%Y-%m')).groupby('month'):report.append({'head':f'onset_h{h}','scope':f'month::{m}',**metrics(x.target.to_numpy(),x.probability.to_numpy())})
 # competing-risk lifecycle: none plus observed archetype, trained only on 2022--25
 causes=train.target__transition_archetype.astype(str).where(train.target__transition_active.eq(1),'none');features=stable;imp=SimpleImputer(strategy='median');x=imp.fit_transform(train[features]);z=imp.transform(test[features]);counts=causes.value_counts();cm=fit(x,causes,seed=99,multi=True,w=causes.map(lambda v:len(causes)/(len(counts)*counts[v])).to_numpy());cp=cm.predict_proba(z);classes=cm.classes_.astype(str);true=test.target__transition_archetype.astype(str).where(test.target__transition_active.eq(1),'none');pred=classes[np.argmax(cp,axis=1)];life_rows=[]
 for scope,g in [('all_2026',pd.DataFrame({'y':true,'p':pred,'month':test.source_utc.dt.strftime('%Y-%m')}))]+[(f'month::{month}',g) for month,g in pd.DataFrame({'y':true,'p':pred,'month':test.source_utc.dt.strftime('%Y-%m')}).groupby('month')]:life_rows.append({'scope':scope,'rows':len(g),'accuracy':accuracy_score(g.y,g.p),'macro_f1':f1_score(g.y,g.p,average='macro',zero_division=0),'causes':len(classes)})
 life=pd.DataFrame(life_rows);cause_rows=[]
 for pos,cause in enumerate(classes):
  q=pd.DataFrame({'source_utc':test.source_utc,'y':(true.to_numpy()==cause).astype(int),'p':cp[:,pos]})
  for scope,g in [('all_2026',q)]+[(f'month::{month}',g) for month,g in q.assign(month=q.source_utc.dt.strftime('%Y-%m')).groupby('month')]:cause_rows.append({'cause':cause,'scope':scope,'support':int(g.y.sum()),'rows':len(g),**metrics(g.y.to_numpy(),g.p.to_numpy())})
 cause_metrics=pd.DataFrame(cause_rows)
 primary=forward[forward.horizon_hours.eq(12)];c=pd.read_parquet(current,columns=['candidate_id','__ts__','execution_net_ev_12h','catboost__residual__without_hpo__all_features']);c.__ts__=pd.to_datetime(c.__ts__,utc=True);c=c[c.__ts__.le(primary.source_utc.max())].copy();c['month']=c.__ts__.dt.strftime('%Y-%m');c['selected']=False
 for _,g in c.groupby('month'):c.loc[g.index,'selected']=global_top10(g,'catboost__residual__without_hpo__all_features')
 e=c[c.selected].merge(primary[['source_utc','probability','target']],left_on='__ts__',right_on='source_utc');e['decile']=pd.qcut(e.probability.rank(method='first'),10,labels=False,duplicates='drop');econ=e.groupby(['month','decile'],as_index=False).agg(rows=('candidate_id','size'),mean_net_bps=('execution_net_ev_12h',lambda x:float(x.mean()*1e4)),mean_probability=('probability','mean'),observed_target=('target','mean'))
 stage=output.parent/f'.{output.name}.{uuid.uuid4().hex}.stage';stage.mkdir(parents=True)
 try:
  hpo.to_csv(stage/'train_only_multihorizon_hpo.csv',index=False);pd.DataFrame(winners).to_csv(stage/'frozen_horizon_winners.csv',index=False);pd.DataFrame(report).to_csv(stage/'untouched_2026_horizon_metrics.csv',index=False);life.to_csv(stage/'untouched_2026_competing_risk_lifecycle.csv',index=False);cause_metrics.to_csv(stage/'untouched_2026_competing_risk_cause_metrics.csv',index=False);econ.to_csv(stage/'global_top10_economic_attribution.csv',index=False);forward.to_parquet(stage/'forward_multihorizon_predictions.parquet',index=False);(stage/'stable_features.json').write_text(json.dumps(stable,indent=2)+'\n');pd.DataFrame([{'split':'train_2022_25','rows':len(train),'active_events':int(train.target__transition_active.sum())},{'split':'untouched_2026','rows':len(test),'active_events':int(test.target__transition_active.sum())}]+[{'split':f'onset_h{h}','rows':len(test),'active_events':int(test[f'target_h{h}'].sum())} for h in HORIZONS]).to_csv(stage/'support.csv',index=False)
  man={'schema':'strict_transition_v3_multihorizon_competing_risk_v2','research_only':True,'promotion_eligible':False,'selection_contract':'feature family, positive class weight, calibration and horizon winner selected only by 2022-25 blocked CV AP-Brier with fold stability; identical untouched 2026 test','separation':'state and ex-post phase excluded from features; lifecycle/archetype are targets only','inputs_sha256':{'catalogue':sha256(catalogue),'current':sha256(current)},'outputs_sha256':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()},'counts':{'train':len(train),'test':len(test),'stable_features':len(stable)}};(stage/'manifest.json').write_text(json.dumps(safe(man),indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(f"{sha256(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--catalogue',type=Path,default=CATALOGUE);p.add_argument('--current',type=Path,default=CURRENT);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();print(json.dumps(safe(run(catalogue=a.catalogue,current=a.current,output=a.output)),sort_keys=True))
