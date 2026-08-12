#!/usr/bin/env python3
"""R3 base + three-class under/accurate/overconfident meta layer.

The target is the side-local, training-only tercile of
``r3_metric_target - (P(clear)-P(adverse))``.  It is a reliability/residual
head, rather than a second model attempting to relearn the R3 event target.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.config import CFG  # noqa:E402
from extreme_price_movements.transport_supervised_archetypes import configured_available_meta_features,training_univariate_screen  # noqa:E402

SOURCE=ROOT/'data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet'
OUT=ROOT/'data_perp/artifacts/r3_base_error_tercile_meta_20260803_v1'
BASE=('r3_meta_p_adverse','r3_meta_p_weak','r3_meta_p_clear','r3_meta_opportunity_score','base_r3_entropy','base_r3_top2_margin')

def _matrix(train,test,fields):
    med=train.loc[:,fields].replace([np.inf,-np.inf],np.nan).median().fillna(0.)
    return train.loc[:,fields].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(np.float32),test.loc[:,fields].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(np.float32)

def _rows(frame,score,arm,fold):
  o=frame.sort_values([score,'candidate_id'],ascending=[False,True],kind='stable');out=[]
  for top in (.01,.05,.10):
   chosen=o.head(max(1,int(np.ceil(len(o)*top))))
   for scope,part in (('global',chosen),('long',chosen[chosen.side_name.eq('long')]),('short',chosen[chosen.side_name.eq('short')])):
    if len(part):out.append({'fold':fold,'arm':arm,'scope':scope,'top_fraction':top,'rows':len(part),'net_bps':float(part.exact_net_bps.mean()),'gross_bps':float(part.exact_gross_bps.mean()),'r3_metric_ic':float(spearmanr(frame[score],frame.r3_metric_target).statistic),'long_share':float(chosen.side_name.eq('long').mean())})
  return out

def _mapping(train):
  # Train-only quantiles are computed separately per side. Class 0 = base
  # overconfident, 1 = relatively accurate, 2 = base underconfident.
  result={}
  for side in ('long','short'):
   residual=(train.loc[train.side_name.eq(side),'r3_metric_target'].to_numpy(float)-train.loc[train.side_name.eq(side),'r3_meta_opportunity_score'].to_numpy(float))
   edges=np.quantile(residual,[1/3,2/3]);labels=np.digitize(residual,edges,right=True);means=np.array([residual[labels==k].mean() for k in range(3)])
   result[side]=(edges,means)
  return result

def run():
 OUT.mkdir(parents=True,exist_ok=True);frame=pd.read_parquet(SOURCE);frame['__ts__']=pd.to_datetime(frame.__ts__,utc=True);frame['label_available_ts']=pd.to_datetime(frame.label_available_ts,utc=True)
 p=frame[['r3_meta_p_adverse','r3_meta_p_weak','r3_meta_p_clear']].to_numpy(float);frame['base_r3_entropy']=-(p*np.log(np.maximum(p,1e-12))).sum(1);q=np.sort(p,axis=1);frame['base_r3_top2_margin']=q[:,-1]-q[:,-2]
 available=configured_available_meta_features(CFG,frame.columns.tolist());coverage=1-frame.loc[:,available].isna().mean();usable=coverage[coverage.ge(.90)].index.tolist();pd.DataFrame({'feature':available,'coverage':coverage.reindex(available),'usable':pd.Index(available).isin(usable)}).to_parquet(OUT/'meta_feature_coverage.parquet',index=False)
 predictions=[];metrics=[];economics=[]
 for fold in (3,4):
  test=frame.loc[frame.fold.eq(fold)].copy();start=test.__ts__.min();train=frame.loc[frame.fold.lt(fold)&frame.label_available_ts.lt(start)].copy();mapping=_mapping(train);scored=[]
  for side in ('long','short'):
   tr=train.loc[train.side_name.eq(side)].copy();te=test.loc[test.side_name.eq(side)].copy();edges,means=mapping[side];residual=tr.r3_metric_target.to_numpy(float)-tr.r3_meta_opportunity_score.to_numpy(float);y=np.digitize(residual,edges,right=True)
   context=training_univariate_screen(tr,usable,residual,maximum=48);fields=[*BASE,*context];xtr,xte=_matrix(tr,te,fields);counts=np.bincount(y,minlength=3).astype(float);w=np.sqrt(len(y)/np.maximum(3*counts[y],1.));w=np.clip(w/w.mean(),.5,2.)
   model=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=140,learning_rate=.04,num_leaves=20,min_child_samples=350,colsample_bytree=.8,reg_lambda=30.,random_state=20260803+fold,n_jobs=1,verbosity=-1).fit(xtr,y,sample_weight=w);prob=np.clip(model.predict_proba(xte),1e-6,1.);prob/=prob.sum(1,keepdims=True);correction=prob@means;z=te.copy();z[['meta_p_base_overconfident','meta_p_base_accurate','meta_p_base_underconfident']]=prob;z['r3_base_error_correction']=correction;z['r3_base_plus_meta_score']=z.r3_meta_opportunity_score.to_numpy(float)+correction;scored.append(z)
   observed=np.digitize(te.r3_metric_target.to_numpy(float)-te.r3_meta_opportunity_score.to_numpy(float),edges,right=True);metrics.append({'fold':fold,'side_name':side,'train_rows':len(tr),'test_rows':len(te),'feature_count':len(fields),'selected_context_features':context,'overconfidence_edge':float(edges[0]),'underconfidence_edge':float(edges[1]),'class_residual_means':means.tolist(),'test_log_loss':float(log_loss(observed,prob,labels=[0,1,2])),'test_accuracy':float(accuracy_score(observed,prob.argmax(1))),'base_r3_ic':float(spearmanr(te.r3_meta_opportunity_score,te.r3_metric_target).statistic),'corrected_r3_ic':float(spearmanr(z.r3_base_plus_meta_score,te.r3_metric_target).statistic)})
  out=pd.concat(scored,ignore_index=True);predictions.append(out);economics.extend(_rows(out,'r3_meta_opportunity_score','R3_base',fold));economics.extend(_rows(out,'r3_base_plus_meta_score','R3_base_plus_error_tercile_meta',fold))
 output=pd.concat(predictions,ignore_index=True);output.to_parquet(OUT/'r3_base_error_tercile_oof_predictions.parquet',index=False);pd.DataFrame(metrics).to_parquet(OUT/'r3_base_error_tercile_metrics.parquet',index=False);pd.DataFrame(economics).to_parquet(OUT/'r3_base_error_tercile_economics.parquet',index=False)
 (OUT/'run_manifest.json').write_text(json.dumps({'schema':'r3_base_error_tercile_meta_v1','base':'direct strict-OOF same-side R3 TP6/SL4 simplex','meta_target':'side-local training terciles of r3_metric_target - direct base opportunity score','classes':{'0':'base_overconfident','1':'relatively_accurate','2':'base_underconfident'},'meta_output':'class probabilities reconstructed to a training-only native-R3 residual correction','no_bps_conversion':True,'strict_label_availability':'meta train labels resolve before test fold start','ranking':'global top-k after direct R3 score + correction','status':'COMPLETED_DIAGNOSTIC_NO_PROMOTION'},indent=2)+'\n')

if __name__=='__main__':run()
