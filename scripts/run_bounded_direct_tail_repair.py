#!/usr/bin/env python3
"""Bounded March-development/April-confirmation direct-tail repair."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile,math
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet';OUT=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v1'
ID=('candidate_id','side_name','__symbol__','__ts__');F=('score_base_alpha','score_residual_expected_ev','direct_q25_return');Y='execution_net_ev_12h';END='execution_label_end_utc';TIME='execution_decision_utc';ARMS=('incumbent_direct_q25','tail_weighted_direct','robust_decomposed','residual_x_conversion_interaction')
def hs(p):
 d=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def confirmation_eligible(frame,start):
 z=frame.loc[pd.to_datetime(frame[END],utc=True,errors='raise').lt(pd.Timestamp(start))].copy()
 if len(z) and pd.to_datetime(z[END],utc=True).max()>=pd.Timestamp(start):raise RuntimeError('availability cutoff')
 return z
def wj(p,x):
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str)+'\n');os.replace(t,p)
def order(x,s,f):
 n=max(1,math.ceil(len(x)*f));z=x.copy()
 for c in ID:z[c]=z[c].astype(str)
 return z.sort_values([s,'candidate_id','__ts__','__symbol__','side_name'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n]
def model(train,valid,arm):
 X=train[list(F)];V=valid[list(F)];base=valid.direct_q25_return.to_numpy(float)
 if arm=='incumbent_direct_q25':return base
 if arm=='residual_x_conversion_interaction':return base # replaced after dev scale freeze
 q1,q2=train.score_base_alpha.quantile(.7),train.score_residual_expected_ev.quantile(.7)
 sw=np.where((train.score_base_alpha>=q1)&(train.score_residual_expected_ev>=q2),2.,1.)
 if arm=='tail_weighted_direct':
  m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=17).fit(X,train[Y],sample_weight=sw);return m.predict(V)
 pos=train[Y].gt(0).to_numpy();
 c=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=19).fit(X,pos,sample_weight=sw)
 pp=c.predict_proba(V)[:,1]
 def cond(mask,val,seed):
  if mask.sum()<100:return np.repeat(float(np.mean(val[mask])) if mask.sum() else 0.,len(valid))
  return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=seed).fit(X.loc[mask],val[mask]).predict(V)
 gain=cond(pos,train[Y].clip(lower=0).to_numpy(),23);loss=cond(~pos,(-train[Y]).clip(lower=0).to_numpy(),29)
 return pp*np.maximum(gain,0)-(1-pp)*np.maximum(loss,0)
def causal_map(dev,score,evals):
 # all calibration source rows are earlier development OOF; no confirmation labels.
 return IsotonicRegression(out_of_bounds='clip').fit(score,dev[Y]).predict(evals)
def met(x,s,month,arm,stage):
 rows=[]
 for f in (.01,.05,.1,.2):
  z=order(x,s,f);net=z[Y];opp=z.opportunity_gross_above_cost_0bps.astype(bool);allp=x.opportunity_gross_above_cost_0bps.astype(bool).sum();cut=z[s].iloc[-1]
  rows.append({'month':month,'arm':arm,'stage':stage,'top_fraction':f,'rows':len(z),'net_bps':float(net.mean()*1e4),'positive_rate':float(net.gt(0).mean()),'opportunity_precision':float(opp.mean()),'opportunity_recall':float(opp.sum()/allp),'full_ic':float(x[s].corr(x[Y],method='spearman')),'tail_ic':float(z[s].corr(net,method='spearman')),'gross_bps':float(z.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(z.execution_cost_return.mean()*1e4),'mfe_bps':float(z.execution_mfe_return_12h.mean()*1e4),'mae_bps':float(z.execution_mae_return_12h.mean()*1e4),'timeout_rate':float(z.execution_exit_reason.astype(str).eq('timeout').mean()),'full_stop_rate':float(z.execution_exit_reason.astype(str).isin(['full_stop','full_sl']).mean()),'long_share':float(z.side_name.eq('long').mean()),'cutoff_tie_rows':int(np.isclose(x[s],cut,rtol=0,atol=1e-14).sum()),'score_distinct':int(x[s].nunique())})
 return rows
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x=pd.read_parquet(a.source);x[TIME]=pd.to_datetime(x[TIME],utc=True);x[END]=pd.to_datetime(x[END],utc=True)
 if x.duplicated(list(ID)).any() or x[list(F)+[Y]].isna().any().any():raise RuntimeError('exact score/label contract unavailable')
 dev=x[x.candidate_month.eq('2025-03')].copy().reset_index(drop=True);conf=x[x.candidate_month.eq('2025-04')].copy().reset_index(drop=True)
 conf_start=conf[TIME].min();dev=confirmation_eligible(dev,conf_start).reset_index(drop=True)
 if dev.empty or dev[END].max()>=conf_start:raise RuntimeError('confirmation fit/calibration availability violates April cutoff')
 # Strict development OOF: three chronological blocks and per-side fits.
 days=np.array(sorted(dev[TIME].dt.floor('D').unique()));starts=[days[int(len(days)*q)] for q in (.4,.6,.8)];o={k:np.full(len(dev),np.nan) for k in ARMS}
 for start in starts:
  vi=(dev[TIME]>=start)&(dev[TIME]<start+pd.Timedelta(days=6));ti=(dev[TIME]<start)&(dev[END]<start)
  for side in ('long','short'):
   tr=dev[ti&dev.side_name.eq(side)];va=dev[vi&dev.side_name.eq(side)]
   for arm in ARMS:o[arm][va.index.to_numpy()-dev.index.min()]=model(tr,va,arm)
 # interaction scale predeclared bounded grid chosen on development OOF only.
 valid=np.isfinite(o['incumbent_direct_q25']); grid=(0.,.25,.5); ranks=[]
 for k in grid:
  ss=o['incumbent_direct_q25'][valid]+k*dev.loc[valid,'score_residual_delta_ev'].to_numpy(float);z=order(dev.loc[valid].assign(s=ss),'s',.1);ranks.append((float(z[Y].mean()),k))
 scale=max(ranks)[1];o['residual_x_conversion_interaction'][valid]=o['incumbent_direct_q25'][valid]+scale*dev.loc[valid,'score_residual_delta_ev'].to_numpy(float)
 # Confirmation: all March labels resolved before April; fit per side once, then map only from dev OOF.
 pred={k:np.full(len(conf),np.nan) for k in ARMS}
 for side in ('long','short'):
  tr=dev[dev.side_name.eq(side)];va=conf[conf.side_name.eq(side)]
  ix=va.index.to_numpy()-conf.index.min()
  for arm in ARMS:pred[arm][ix]=model(tr,va,arm)
  pred['residual_x_conversion_interaction'][ix]=va.direct_q25_return.to_numpy(float)+scale*va.score_residual_delta_ev.to_numpy(float)
 rows=[];parts=[]
 for arm in ARMS:
  conf[arm]=pred[arm];dev[arm]=o[arm]
  d=dev[np.isfinite(dev[arm])].copy();mapped=causal_map(d,d[arm].to_numpy(float),conf[arm].to_numpy(float));conf['map_'+arm]=mapped
  for m,z in [('2025-03',dev[np.isfinite(dev[arm])]),('2025-04',conf),('pooled_confirmation',conf)]:
   if m=='pooled_confirmation':z=conf
   rows+=met(z,arm,m,arm,'raw')
   if m!='2025-03':rows+=met(z,'map_'+arm,m,arm,'causal_map')
  parts.append(conf[list(ID)+[TIME,Y,'candidate_month',arm,'map_'+arm]].assign(arm=arm))
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));paths={'metrics':st/'metrics.csv','predictions':st/'confirmation_predictions.parquet','development_scale':st/'development_scale.json'}
 pd.DataFrame(rows).to_csv(paths['metrics'],index=False);pd.concat(parts).to_parquet(paths['predictions'],index=False);wj(paths['development_scale'],{'grid':grid,'development_oof_only':True,'scores':[(v,k) for v,k in ranks],'frozen_scale':scale})
 man={'schema':'bounded_direct_tail_repair_v2','status':'COMPLETED_NONPROMOTION','promotion_eligible':False,'contract':{'split':'March strict chronological OOF development; April one untouched confirmation','confirmation_start_utc':str(conf_start),'max_fit_or_calibration_label_end_utc':str(dev[END].max()),'arms':ARMS,'features':'frozen base/residual/direct score lineages only; per-side fits','mapping':'isotonic fit on March OOF only whose label ends strictly before confirmation start','selection':'pooled-global candidate-ID ties; month/side are diagnostics','actions':'timing/MAE/target/wait excluded','policy':'not run'},'input':{'path':str(a.source),'sha256':hs(a.source)},'frozen_interaction_scale':scale,'outputs':{k:{'path':str(a.output_dir/v.name),'sha256':hs(v)} for k,v in paths.items()}}
 wj(st/'manifest.json',man);(st/'manifest.sha256').write_text(hs(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
def parser():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SRC);p.add_argument('--output-dir',type=Path,required=True);return p
if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2))
