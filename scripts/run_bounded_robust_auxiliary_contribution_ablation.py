#!/usr/bin/env python3
"""Corrected strict March-OOF / April robust-control support-head screen.

The control exactly reconstructs the v2 ``robust_decomposed`` score.  Weight
selection uses raw March OOF top-10 only.  A map is fitted once on that OOF
ledger and used exclusively for April; it is never scored on its own fit rows.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet'
V2=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v2'
PEAK=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2/oof_predictions.parquet'
SLOPE=ROOT/'data_perp/artifacts/febapr2025_historical_future_slope_fixed_geometry_oof_20260730_v1/oof_predictions.parquet'
ID=('candidate_id','side_name','__symbol__','__ts__'); F=('score_base_alpha','score_residual_expected_ev','direct_q25_return'); Y='execution_net_ev_12h'; END='execution_label_end_utc'; TIME='execution_decision_utc'; FRACS=(.01,.05,.1,.2); WEIGHTS=(0.,.1,.25); ARMS=('control','peak_contribution','future_slope','both')
def hs(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as h:
  for b in iter(lambda:h.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def wj(p,x):
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(t,p)
def order(x,s,f):
 n=max(1,math.ceil(len(x)*f));return x.sort_values([s,'candidate_id','__ts__','__symbol__','side_name'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n].copy()
def robust(train,valid):
 X=train[list(F)]; V=valid[list(F)]; pos=train[Y].gt(0).to_numpy()
 # Exact v2 recipe: the robust hurdle classifier retains the v2 tail weights;
 # its conditional gain/loss regressors are deliberately unweighted.
 q1,q2=train.score_base_alpha.quantile(.7),train.score_residual_expected_ev.quantile(.7);sw=np.where((train.score_base_alpha>=q1)&(train.score_residual_expected_ev>=q2),2.,1.)
 c=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=19).fit(X,pos,sample_weight=sw)
 pp=c.predict_proba(V)[:,1]
 def cond(mask,val,seed):
  if mask.sum()<100:return np.repeat(float(np.mean(val[mask])) if mask.sum() else 0.,len(valid))
  return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=seed).fit(X.loc[mask],val[mask]).predict(V)
 gain=cond(pos,train[Y].clip(lower=0).to_numpy(),23); loss=cond(~pos,(-train[Y]).clip(lower=0).to_numpy(),29)
 return pp*np.maximum(gain,0)-(1-pp)*np.maximum(loss,0)
def reconstruct(x):
 dev=x[x.candidate_month.eq('2025-03')].copy().reset_index(drop=True); conf=x[x.candidate_month.eq('2025-04')].copy().reset_index(drop=True); start=conf[TIME].min()
 dev=dev[dev[END].lt(start)].copy().reset_index(drop=True)
 if dev.empty or dev[END].max()>=start:raise ValueError('March availability cutoff violation')
 days=np.array(sorted(dev[TIME].dt.floor('D').unique())); starts=[days[int(len(days)*q)] for q in (.4,.6,.8)]
 o=np.full(len(dev),np.nan)
 for cut in starts:
  vi=(dev[TIME]>=cut)&(dev[TIME]<cut+pd.Timedelta(days=6)); ti=(dev[TIME]<cut)&(dev[END]<cut)
  for side in ('long','short'):
   tr=dev[ti&dev.side_name.eq(side)]; va=dev[vi&dev.side_name.eq(side)]
   if len(va):o[va.index.to_numpy()]=robust(tr,va)
 pred=np.full(len(conf),np.nan)
 for side in ('long','short'):
  tr=dev[dev.side_name.eq(side)];va=conf[conf.side_name.eq(side)];pred[va.index.to_numpy()]=robust(tr,va)
 dev['robust_decomposed']=o;conf['robust_decomposed']=pred
 return dev,conf,starts
def zfit(ref,val):
 med=float(ref.median()); sd=float(ref.std(ddof=0));sd=sd if np.isfinite(sd) and sd>1e-12 else 1.;return (val.to_numpy(float)-med)/sd,{'median':med,'std':sd}
def ece(pred,y,bins=10):
 if len(pred)==0:return np.nan
 q=np.linspace(0,1,bins+1); edges=np.unique(np.quantile(pred,q));
 if len(edges)<2:return float(abs(np.mean(pred)-np.mean(y)))
 total=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(pred>=lo)&((pred<hi)|(hi==edges[-1]));total+=m.mean()*abs(pred[m].mean()-y[m].mean()) if m.any() else 0.
 return float(total)
def metrics(x,arm,weight,stage,month):
 rows=[];sides=[];assets=[]
 for kind,col in [('raw','raw_score'),('mapped','mapped_score')]:
  for f in FRACS:
   q=order(x,col,f);cut=float(q[col].iloc[-1]);tie=int(np.isclose(x[col].to_numpy(float),cut,rtol=0,atol=1e-14).sum());p=q[col].to_numpy(float);y=q[Y].to_numpy(float)
   rows.append({'arm':arm,'weight':weight,'stage':stage,'month':month,'score_kind':kind,'top_fraction':f,'rows':len(q),'net_bps':float(y.mean()*1e4),'gross_bps':float(q.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(q.execution_cost_return.mean()*1e4),'positive_rate':float((y>0).mean()),'full_rank_ic':float(x[col].corr(x[Y],method='spearman')),'cutoff':cut,'cutoff_tie_rows':tie,'cutoff_tie_fraction_of_book':float(tie/len(q)),'prediction_bias_bps':float((p-y).mean()*1e4),'prediction_mae_bps':float(np.abs(p-y).mean()*1e4),'calibration_ece_bps':float(ece(p,y)*1e4),'latest_fold_coverage':month=='2025-04','selection':'one_global_top_k_stable_candidate_id_ties'})
   for side,v in q.groupby('side_name',sort=True):sides.append({'arm':arm,'weight':weight,'stage':stage,'month':month,'score_kind':kind,'top_fraction':f,'side_name':side,'rows':len(v),'share':float(len(v)/len(q)),'net_bps':float(v[Y].mean()*1e4),'positive_rate':float(v[Y].gt(0).mean())})
   for asset,v in q.groupby('__symbol__',sort=True):assets.append({'arm':arm,'weight':weight,'stage':stage,'month':month,'score_kind':kind,'top_fraction':f,'__symbol__':asset,'rows':len(v),'share':float(len(v)/len(q)),'net_bps':float(v[Y].mean()*1e4)})
 return rows,sides,assets
def load(a):
 x=pd.read_parquet(a.source);p=pd.read_parquet(a.peak,columns=list(ID)+['pred_peak_mfe_12h_atr__p_hit','pred_peak_mfe_12h_atr__conditional_mean']);s=pd.read_parquet(a.slope,columns=list(ID)+['pred_future_slope_atr_per_hour__diagnostic'])
 for z in (x,p,s):z['__ts__']=pd.to_datetime(z['__ts__'],utc=True);assert not z.duplicated(list(ID)).any()
 x=x.merge(p,on=list(ID),validate='one_to_one').merge(s,on=list(ID),validate='one_to_one');x[TIME]=pd.to_datetime(x[TIME],utc=True);x[END]=pd.to_datetime(x[END],utc=True)
 if len(x)!=140682 or not np.allclose(x.execution_gross_ev_12h-x.execution_cost_return,x[Y],atol=1e-12,rtol=0):raise ValueError('common population/gross-net contract')
 return x
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x=load(a);dev,conf,cuts=reconstruct(x)
 prior=pd.read_parquet(a.v2/'confirmation_predictions.parquet'); prior=prior[prior.arm.eq('robust_decomposed')][list(ID)+['robust_decomposed']].rename(columns={'robust_decomposed':'v2_robust'})
 parity=conf.merge(prior,on=list(ID),validate='one_to_one'); delta=np.abs(parity.robust_decomposed-parity.v2_robust);assert len(parity)==len(conf) and float(delta.max())==0.,'April robust control must bit-match v2'
 dev=dev[np.isfinite(dev.robust_decomposed)].copy();
 for z in (dev,conf):z['peak_contribution']=z.pred_peak_mfe_12h_atr__p_hit*z.pred_peak_mfe_12h_atr__conditional_mean
 scales={};dev['base_z'],scales['robust_decomposed']=zfit(dev.robust_decomposed,dev.robust_decomposed);conf['base_z'],_=zfit(dev.robust_decomposed,conf.robust_decomposed)
 for col in ('peak_contribution','pred_future_slope_atr_per_hour__diagnostic'):
  dev[col+'_z'],scales[col]=zfit(dev[col],dev[col]);conf[col+'_z'],_=zfit(dev[col],conf[col])
 choices=[];met=[];si=[];aa=[];led=[]
 add={'control':(), 'peak_contribution':('peak_contribution_z',), 'future_slope':('pred_future_slope_atr_per_hour__diagnostic_z',), 'both':('peak_contribution_z','pred_future_slope_atr_per_hour__diagnostic_z')}
 base_scale=scales['robust_decomposed']['std']
 for arm in ARMS:
  for weight in WEIGHTS:
   if arm=='control' and weight:continue
   d=dev.copy();c=conf.copy();d['raw_score']=d.robust_decomposed+weight*base_scale*sum((d[v] for v in add[arm]),start=pd.Series(0.,index=d.index));c['raw_score']=c.robust_decomposed+weight*base_scale*sum((c[v] for v in add[arm]),start=pd.Series(0.,index=c.index))
   # This map is never evaluated on March: raw March OOF is the sole selector.
   mapper=IsotonicRegression(out_of_bounds='clip').fit(d.raw_score,d[Y]);c['mapped_score']=mapper.predict(c.raw_score)
   top=order(d,'raw_score',.1);choices.append({'arm':arm,'weight':weight,'march_oof_raw_top10_net_bps':float(top[Y].mean()*1e4),'rows':len(top)})
   for stage,month,z in [('development_oof_raw_only','2025-03',d.assign(mapped_score=np.nan)),('confirmation','2025-04',c)]:
    if stage.startswith('development'): # report raw only; a fitted map cannot be evaluated here
     r,ss,az=metrics(z.assign(mapped_score=z.raw_score),arm,weight,stage,month);r=[v for v in r if v['score_kind']=='raw'];ss=[v for v in ss if v['score_kind']=='raw'];az=[v for v in az if v['score_kind']=='raw']
    else:r,ss,az=metrics(z,arm,weight,stage,month)
    met+=r;si+=ss;aa+=az
   led.append(c[list(ID)+[TIME,END,Y,'robust_decomposed','raw_score','mapped_score']].assign(arm=arm,weight=weight))
 choice=pd.DataFrame(choices).sort_values(['march_oof_raw_top10_net_bps','arm','weight'],ascending=[False,True,True],kind='mergesort');win=choice.iloc[0].to_dict()
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));pd.DataFrame(met).to_csv(st/'global_metrics.csv',index=False);pd.DataFrame(si).to_csv(st/'side_metrics.csv',index=False);pd.DataFrame(aa).to_csv(st/'asset_metrics.csv',index=False);choice.to_csv(st/'march_oof_raw_weight_selection.csv',index=False);pd.concat(led,ignore_index=True).to_parquet(st/'april_confirmation_predictions.parquet',index=False,compression='zstd');wj(st/'control_parity.json',{'v2_confirmation_rows':len(prior),'reconstructed_rows':len(conf),'max_abs_delta':float(delta.max()),'bit_identical':True,'v2_manifest_sha256':hs(a.v2/'manifest.json')})
 outs={p.name:hs(p) for p in st.iterdir() if p.is_file()};man={'schema':'bounded_robust_auxiliary_contribution_ablation_v2','status':'COMPLETED_RESEARCH_ONLY_NO_PORTFOLIO_REPLAY','promotion_eligible':False,'contract':{'control':'exact reconstructed bounded_direct_tail_repair_v2 robust_decomposed; April raw score bit-identical','March_OOF':'strict chronological v2 blocks; raw score only for weight selection','map':'fit once on eligible March OOF labels, applied only to April; never evaluated on map fit labels','confirmation':'April untouched labels','selection':'one pooled global top K with stable candidate-ID ties; sides/assets attribution only','arms':ARMS,'weights':WEIGHTS,'peak_formula':'P(hit)*E(peak ATR|hit)','slope':'strict OOF prediction only; realised slope forbidden','actions':'excluded','portfolio_replay':'NOT_RUN'},'input_sha256':{str(p):hs(p) for p in (a.source,a.v2/'manifest.json',a.peak,a.slope)},'march_oof_blocks_utc':[str(x) for x in cuts],'frozen_component_scales':scales,'frozen_april_winner_from_raw_march_oof':win,'outputs_sha256':outs,'runner_sha256':hs(Path(__file__))};wj(st/'manifest.json',man);(st/'manifest.sha256').write_text(hs(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SRC);p.add_argument('--v2',type=Path,default=V2);p.add_argument('--peak',type=Path,default=PEAK);p.add_argument('--slope',type=Path,default=SLOPE);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,default=str))
