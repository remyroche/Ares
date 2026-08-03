#!/usr/bin/env python3
"""Paired, causal mapping-only A5 versus residual-control comparison.

No predictive model is fitted here.  Every mapper is refit daily on only
resolved 21-day labels from the sealed v2 score ledger.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data_perp/artifacts/canonical_execution_reliability_ablation_20260730_v2'
DEFAULT=ROOT/'data_perp/artifacts/canonical_execution_reliability_mapping_ablation_20260730_v2'
V1=ROOT/'data_perp/artifacts/canonical_execution_reliability_mapping_ablation_20260730_v1'
TIME,END,NET,GROSS,COST='execution_decision_utc','execution_label_end_utc','execution_net_ev_12h','execution_gross_ev_12h','execution_cost_return'
ID=('candidate_id','side_name','__symbol__','__ts__'); A5='A5__A2__context__timestamp_side_relative'; RES='A0__score_residual_expected_ev'
WINDOW=pd.Timedelta(days=21);POOL,SIDE,LAMBDA=2000,1000,1000.;TOPS=(.01,.05,.1,.2);APR0=pd.Timestamp('2025-04-01T00:00:00Z');APR1=pd.Timestamp('2025-05-01T00:00:00Z');L7=pd.Timestamp('2025-04-24T00:00:00Z')
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p:Path,x:Any)->None:
 q=p.with_name('.'+p.name+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def stable_order(x:pd.DataFrame,col:str)->pd.DataFrame:return x.sort_values([col,'candidate_id','side_name','__symbol__','__ts__'],ascending=[False,True,True,True,True],kind='mergesort')
def strict_snapshot(raw:np.ndarray,value:np.ndarray)->np.ndarray:
 """PAVA + infinitesimal rank breaks: strictly preserves raw order per snapshot."""
 order=np.argsort(raw,kind='mergesort'); x=raw[order]; y=value[order]; z=IsotonicRegression(increasing=True,out_of_bounds='clip').fit_transform(x,y); eps=max(1e-15,float(np.nanmax(np.abs(z)))*1e-12);z=z+eps*np.arange(len(z));out=np.empty(len(z));out[order]=z;return out
def iso(h:pd.DataFrame)->IsotonicRegression:return IsotonicRegression(out_of_bounds='clip').fit(h.raw_score,h[NET])
def positive_huber(h:pd.DataFrame,raw:np.ndarray)->np.ndarray:
 x=h.raw_score.to_numpy(float);y=h[NET].to_numpy(float);m=np.median(x);s=np.median(np.abs(x-m)) or 1.;z=np.tanh((x-m)/s);X=np.c_[np.ones(len(z)),z];w=np.ones(len(z));b=np.zeros(2)
 for _ in range(20):
  b=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0];r=y-X@b;scale=max(np.median(np.abs(r))/0.6745,1e-6);w=np.minimum(1.,1.5*scale/np.maximum(np.abs(r),1e-12))
 # Fixed positive-slope constraint is the point of M2; no HPO/sign choice.
 slope=max(float(b[1]),1e-15);return float(b[0])+slope*np.tanh((raw-m)/s)
def m3_bins(h:pd.DataFrame,e:pd.DataFrame)->np.ndarray:
 """Pooled-only 20 equal percentile bins, pseudo-count 200, strict PAVA."""
 hp=h.groupby(TIME,sort=False).raw_score.rank(pct=True,method='first').to_numpy(float);prior=float(h[NET].mean());edges=np.linspace(0.,1.,21);centers=(edges[:-1]+edges[1:])/2
 bucket=np.minimum((hp*20).astype(int),19);count=np.bincount(bucket,minlength=20);total=np.bincount(bucket,weights=h[NET].to_numpy(float),minlength=20);vals=(total+200*prior)/(count+200)
 vals=strict_snapshot(centers,np.asarray(vals)); ep=e.groupby(TIME,sort=False).raw_score.rank(pct=True,method='first').to_numpy(float);return np.interp(ep,centers,vals,left=vals[0],right=vals[-1])
def history(x:pd.DataFrame,s:pd.Timestamp)->pd.DataFrame:
 h=x[x[END].ge(s-WINDOW)&x[END].lt(s)&x.score_available_utc.lt(s)].copy()
 if h.duplicated(list(ID)).any():raise RuntimeError('reference identities duplicate')
 return h
def map_day(h:pd.DataFrame,e:pd.DataFrame)->tuple[pd.DataFrame,dict]:
 s=e[TIME].dt.floor('D').iloc[0];o=e[list(ID)].copy();methods=['baseline','M1_strict_pooled','M2_positive_huber','M3_pooled_timestamp_pct']
 for c in methods:o[c]=np.nan
 o['eligible']=False;overlap=len(set(e.candidate_id.astype(str)) & set(h.candidate_id.astype(str)));audit={'snapshot_utc':s,'evaluation_rows':len(e),'reference_rows':len(h),'reference_min_label_end_utc':h[END].min() if len(h) else pd.NaT,'reference_max_label_end_utc':h[END].max() if len(h) else pd.NaT,'causal_window_exact':bool(h[END].ge(s-WINDOW).all() and h[END].lt(s).all()),'reference_score_available_before_snapshot':bool(h.score_available_utc.lt(s).all()),'evaluation_reference_id_overlap':overlap,'pooled_support_pass':len(h)>=POOL and h.raw_score.nunique()>1}
 if overlap:raise RuntimeError('evaluation/reference overlap')
 if not audit['pooled_support_pass']:return o,audit
 raw=e.raw_score.to_numpy(float);pool=iso(h);p=pool.predict(raw);o['baseline']=p
 for side in ('long','short'):
  bm=e.side_name.eq(side).to_numpy();hs=h[h.side_name.eq(side)];audit[side+'_reference_rows']=len(hs)
  if len(hs)>=SIDE and hs.raw_score.nunique()>1:
   w=len(hs)/(len(hs)+LAMBDA);o.loc[bm,'baseline']=p[bm]+w*(iso(hs).predict(raw[bm])-pool.predict(raw[bm]));audit[side+'_weight']=w
  else:audit[side+'_weight']=0.
 o['M1_strict_pooled']=strict_snapshot(raw,p);o['M2_positive_huber']=strict_snapshot(raw,positive_huber(h,raw));o['M3_pooled_timestamp_pct']=strict_snapshot(raw,m3_bins(h,e));o['eligible']=True;audit['common_mapper_eligible']=True
 for m in ('M1_strict_pooled','M2_positive_huber','M3_pooled_timestamp_pct'):
  inv=0; plateau=False
  for _,idx in e.groupby(TIME,sort=False).groups.items():
   pos=e.index.get_indexer(pd.Index(idx));sorter=np.argsort(e.loc[idx,'raw_score'].to_numpy(float),kind='mergesort');v=o[m].to_numpy(float)[pos][sorter]
   inv += int((np.diff(v)<=0).sum());plateau = plateau or bool(pd.Series(v).duplicated().any())
  audit[m+'_within_snapshot_inversions']=inv;audit[m+'_has_plateau']=plateau
 return o,audit
def load(source:Path)->pd.DataFrame:
 if sha(source/'manifest.json')!=(source/'manifest.sha256').read_text().split()[0]:raise RuntimeError('source manifest seal mismatch')
 sm=json.loads((source/'manifest.json').read_text());
 if sm.get('schema')!='canonical_execution_reliability_ablation_v2':raise RuntimeError('sealed reliability v2 source required')
 cols=[*ID,TIME,END,NET,GROSS,COST,'candidate_month','raw_score','score_available_utc','outer_fold','candidate_score_is_oof','config','regime_execution_risk_quintile','execution_exit_class']
 # Predicate pushdown matters: the sealed ledger contains 21 configurations,
 # while this paired mapping comparison is deliberately restricted to two.
 x=pd.read_parquet(source/'scores.parquet',columns=cols,filters=[('config','in',[A5,RES])]).copy()
 for c in (TIME,END,'score_available_utc','__ts__'):x[c]=pd.to_datetime(x[c],utc=True)
 if x.groupby('config').size().to_dict()!={A5:94602,RES:94602}:raise RuntimeError('paired source config row contract failed')
 a=x[x.config.eq(A5)].sort_values(list(ID));b=x[x.config.eq(RES)].sort_values(list(ID));
 if not a[list(ID)].reset_index(drop=True).equals(b[list(ID)].reset_index(drop=True)):raise RuntimeError('source configs are not exact same cohorts')
 if not np.allclose(x[GROSS]-x[COST],x[NET],atol=1e-12) or not x[END].eq(x[TIME]+pd.Timedelta(hours=12)).all():raise RuntimeError('exact cost/H12 contract failed')
 return x
def map_all(x:pd.DataFrame,start:pd.Timestamp|None=None,end:pd.Timestamp|None=None)->tuple[pd.DataFrame,pd.DataFrame]:
 mapped={};aud={}
 for cfg,d in x.groupby('config',sort=True):
  ps=[];aa=[]
  for day,e in d.groupby(d[TIME].dt.floor('D'),sort=True):
   if (start is not None and day<start) or (end is not None and day>=end):continue
   m,a=map_day(history(d,pd.Timestamp(e[TIME].dt.floor('D').iloc[0])),e.copy());ps.append(m);aa.append(a)
  mapped[cfg]=d.merge(pd.concat(ps,ignore_index=True),on=list(ID),validate='one_to_one');aud[cfg]=pd.DataFrame(aa)
 common=set.intersection(*[set(z.loc[z.pooled_support_pass,'snapshot_utc']) for z in aud.values()])
 out=[]
 for cfg,z in mapped.items():
  z=z[z[TIME].dt.floor('D').isin(common)&z.eligible].copy();z['config']=cfg;out.append(z)
 audit=pd.concat([v.assign(config=k,common_eligible_day=v.snapshot_utc.isin(common)) for k,v in aud.items()],ignore_index=True)
 return pd.concat(out,ignore_index=True),audit
def fractional_book(x:pd.DataFrame,col:str,f:float,sorted_x:pd.DataFrame|None=None)->tuple[pd.Series,dict]:
 s=stable_order(x,col) if sorted_x is None else sorted_x;n=max(1,math.ceil(len(s)*f));cut=float(s[col].iloc[n-1]);above=s[s[col]>cut];tie=s[s[col]==cut];need=n-len(above);w=pd.Series(0.,index=x.index);w.loc[above.index]=1.;w.loc[tie.index]=need/len(tie)
 return w,{'top_fraction':f,'selected_rows':n,'cutoff':cut,'boundary_tie_population':len(tie),'tie_selected_share':need/n,'exact_equality_ties':True}
def economics(x:pd.DataFrame,cfg:str,method:str,stage:str)->list[dict]:
 rows=[];sorted_x=stable_order(x,method)
 for f in TOPS:
  w,m=fractional_book(x,method,f,sorted_x);rows.append({'config':cfg,'method':method,'stage':stage,**m,'candidate_rows':len(x),'expected_net_bps':float((w*x[NET]).sum()/m['selected_rows']*1e4),'gross_bps':float((w*x[GROSS]).sum()/m['selected_rows']*1e4),'cost_bps':float((w*x[COST]).sum()/m['selected_rows']*1e4),'rank_ic':float(x[method].corr(x[NET],method='spearman'))})
 return rows
def attribution(x:pd.DataFrame,cfg:str,method:str,stage:str)->tuple[list[dict],list[dict]]:
 rows=[];recon=[];sorted_x=stable_order(x,method)
 for f in TOPS:
  w,m=fractional_book(x,method,f,sorted_x);total=float((w*x[NET]).sum()/m['selected_rows']*1e4)
  for dim in ('side_name','__symbol__','regime_execution_risk_quintile','execution_exit_class'):
   parts=[]
   for value,idx in x.groupby(dim,dropna=False,sort=True).groups.items():
    v=float((w.loc[idx]*x.loc[idx,NET]).sum()/m['selected_rows']*1e4);parts.append(v);rows.append({'config':cfg,'method':method,'stage':stage,'top_fraction':f,'dimension':dim,'value':str(value),'expected_selected_rows':float(w.loc[idx].sum()),'net_bps_contribution':v,'gross_bps_contribution':float((w.loc[idx]*x.loc[idx,GROSS]).sum()/m['selected_rows']*1e4),'cost_bps_contribution':float((w.loc[idx]*x.loc[idx,COST]).sum()/m['selected_rows']*1e4)})
   recon.append({'config':cfg,'method':method,'stage':stage,'top_fraction':f,'dimension':dim,'global_net_bps':total,'sum_group_net_bps':sum(parts),'absolute_reconciliation_error':abs(total-sum(parts))})
 return rows,recon
def run(source:Path=SOURCE,output_dir:Path=DEFAULT,partials:tuple[Path,...]=())->dict:
 if output_dir.exists():raise FileExistsError(output_dir)
 x=load(source)
 if partials:
  z=pd.concat([pd.read_parquet(p/'mapped.parquet') for p in partials],ignore_index=True);audit=pd.concat([pd.read_parquet(p/'audit.parquet') for p in partials],ignore_index=True)
  if z.duplicated(['config',*ID]).any():raise RuntimeError('partial mapping identities overlap')
 else:z,audit=map_all(x)
 methods=['baseline','M1_strict_pooled','M2_positive_huber','M3_pooled_timestamp_pct'];econ=[];attr=[];recon=[];transport=[]
 for cfg,d in z.groupby('config',sort=True):
  march=d[(d[TIME]<APR0)&d.candidate_month.eq('2025-03')];apr=d[(d[TIME]>=APR0)&(d[TIME]<APR1)];
  for m in methods:
   for stage,p in [('march_aggregate',march),*[(f'march_fold_{f}',march[march.outer_fold.eq(f)]) for f in ('selection_1','selection_2','selection_3')],('april_aggregate',apr),('april_latest7d',apr[(apr[TIME]>=L7)&(apr[TIME]<APR1)])]:econ+=economics(p,cfg,m,stage)
   a,r=attribution(apr,cfg,m,'april_aggregate');attr+=a;recon+=r
   for stage,p in [('march',march),('april',apr)]:
    raww,_=fractional_book(p,'raw_score',.1);mapw,_=fractional_book(p,m,.1);transport += [{'config':cfg,'method':m,'stage':stage,'metric':'raw_map_rank_correlation','value':float(p.raw_score.corr(p[m],method='spearman'))},{'config':cfg,'method':m,'stage':stage,'metric':'raw_map_top10_fractional_overlap','value':float(np.minimum(raww,mapw).sum()/mapw.sum())}]
   prev=None
   for day,p in apr.groupby(apr[TIME].dt.floor('D'),sort=True):
    w,_=fractional_book(p,m,.1);now=set((p.loc[w>0,'side_name'].astype(str)+'|'+p.loc[w>0,'__symbol__'].astype(str)))
    if prev is not None:transport.append({'config':cfg,'method':m,'stage':'april','snapshot_utc':day,'metric':'day_to_day_symbol_side_turnover','value':float(1-len(now&prev)/max(1,len(now|prev)))})
    prev=now
 E=pd.DataFrame(econ);sel=[]
 for cfg in (A5,RES):
  for m in methods:
   q=E[(E.config==cfg)&(E.method==m)&E.stage.str.startswith('march_fold_')&(E.top_fraction==.1)].sort_values('stage');v=q.expected_net_bps.to_numpy();sel.append({'config':cfg,'method':m,'mean_bps':float(v.mean()),'std_bps':float(v.std()),'worst_bps':float(v.min()),'latest_bps':float(v[-1]),'objective':float(v.mean()-.5*v.std()+.25*v.min())})
 S=pd.DataFrame(sel);g=[]
 for m in methods:
  for stage in ('march_aggregate','april_aggregate','april_latest7d'):
   aa=E[(E.config==A5)&(E.method==m)&(E.stage==stage)&(E.top_fraction==.1)].iloc[0];rr=E[(E.config==RES)&(E.method==m)&(E.stage==stage)&(E.top_fraction==.1)].iloc[0];g.append({'method':m,'gate':'A5 minus residual '+stage+' top10 >0','pass':aa.expected_net_bps-rr.expected_net_bps>0,'detail':float(aa.expected_net_bps-rr.expected_net_bps)})
  ss=S[(S.config==A5)&(S.method==m)].iloc[0];a5s=pd.DataFrame(attr);a5s=a5s[(a5s.config==A5)&(a5s.method==m)&(a5s.dimension=='side')&(a5s.top_fraction==.1)];common_audit=audit[audit.common_eligible_day];tm=pd.DataFrame(transport);tm=tm[(tm.config==A5)&(tm.method==m)]
  g += [{'method':m,'gate':'A5 March mean/latest/worst top10 >0','pass':bool(ss[['mean_bps','latest_bps','worst_bps']].gt(0).all()),'detail':json.dumps(ss[['mean_bps','latest_bps','worst_bps']].to_dict())},{'method':m,'gate':'A5 April top10 tie selected share <=5%','pass':bool(aa.tie_selected_share<=.05),'detail':float(aa.tie_selected_share)},{'method':m,'gate':'A5 April both side contributions positive','pass':bool((a5s.net_bps_contribution>0).all()),'detail':json.dumps(dict(zip(a5s.value,a5s.net_bps_contribution)))},{'method':m,'gate':'common causal mapping legal','pass':bool((common_audit.causal_window_exact&common_audit.reference_score_available_before_snapshot&common_audit.evaluation_reference_id_overlap.eq(0)).all()),'detail':int(len(common_audit))},{'method':m,'gate':'transport and turnover finite','pass':bool(np.isfinite(tm.value).all()),'detail':int(len(tm))}]
 G=pd.DataFrame(g);stage=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'.',dir=output_dir.parent))
 try:
  # The paired source-score ledger is already immutable and hash-bound below.
  # Persist audit/economics/books rather than duplicating its 180k raw rows.
  audit.to_csv(stage/'causal_mapping_audit.csv',index=False);E.to_csv(stage/'economics.csv',index=False);S.to_csv(stage/'march_selection.csv',index=False);pd.DataFrame(attr).to_csv(stage/'fractional_global_book_attribution.csv',index=False);pd.DataFrame(recon).to_csv(stage/'attribution_reconciliation.csv',index=False);pd.DataFrame(transport).to_csv(stage/'transport_turnover.csv',index=False);G.to_csv(stage/'promotion_gates.csv',index=False)
  files={p.name:sha(p) for p in stage.iterdir() if p.is_file()};m={'schema':'canonical_execution_reliability_mapping_ablation_v2','status':'SEALED_RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY','promotion_eligible':False,'run_id':'canonical_execution_reliability_mapping_ablation_20260730_v2','source':{'artifact':str(source),'scores_sha256':sha(source/'scores.parquet'),'manifest_sha256':sha(source/'manifest.json'),'configs':[A5,RES],'same_candidate_cohort_asserted':True},'contract':{'dataset':'sealed v2 scores only; no predictive retraining','labels':'exact H12 deployed-policy net EV','mapping':'21d S-21d <= label_end < S; common eligible-day intersection across both configs/mappers','methods':'baseline current pooled+shrunk side; M1 strict pooled per-snapshot; M2 fixed positive-slope Huber/tanh per-snapshot; M3 pooled-only 20 fixed equal-width timestamp-percentile bins, pseudo-count 200, strict PAVA','selection':'fractional exact-equality pooled-global books only; no quotas','april_diagnostic_utc':'[2025-04-01,2025-05-01)','latest7_utc':'[2025-04-24,2025-05-01)','costs':'gross - one cost = net','policy':'research only; no replay','random_seed':'not applicable deterministic maps','purge':'label_end and score availability strictly before snapshot','code_revision':'not available; runner hash sealed'},'constants':{'pooled_min':POOL,'side_min':SIDE,'side_lambda':LAMBDA,'m3_bins':20,'m3_pseudocount':200},'outputs_sha256':files,'runner_sha256':sha(Path(__file__))};write(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(sha(stage/'manifest.json')+'  manifest.json\n');write(stage/'seal.json',{'manifest_sha256':sha(stage/'manifest.json'),'files_sha256':{p.name:sha(p) for p in stage.iterdir() if p.is_file()},'status':m['status']});os.replace(stage,output_dir)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output-dir',type=Path,default=DEFAULT);p.add_argument('--partial',type=Path,action='append',default=[]);a=p.parse_args();print(json.dumps(run(a.source,a.output_dir,tuple(a.partial)),indent=2))
