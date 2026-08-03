#!/usr/bin/env python3
"""Bounded no-retrain causal EV-map comparison over one sealed score ledger."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data_perp/artifacts/canonical_execution_reliability_ablation_20260730_v2'
DEFAULT=ROOT/'data_perp/artifacts/canonical_execution_reliability_mapping_ablation_20260730_v1'
TIME,END,NET,GROSS,COST='execution_decision_utc','execution_label_end_utc','execution_net_ev_12h','execution_gross_ev_12h','execution_cost_return'
ID=('candidate_id','side_name','__symbol__','__ts__')
WINDOW=pd.Timedelta(days=21); POOL,SIDE,LAMBDA=2000,1000,1000.; TOPS=(.01,.05,.1,.2)
CONFIG='A5__A2__context__timestamp_side_relative'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p:Path,x:Any)->None:
 q=p.with_name('.'+p.name+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def order(x:pd.DataFrame,col:str,f:float)->pd.DataFrame:
 return x.sort_values([col,'candidate_id','side_name','__symbol__','__ts__'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:max(1,math.ceil(len(x)*f))]
def current_history(all_rows:pd.DataFrame,snapshot:pd.Timestamp)->pd.DataFrame:
 h=all_rows[all_rows[END].ge(snapshot-WINDOW)&all_rows[END].lt(snapshot)&all_rows.score_available_utc.lt(snapshot)].copy()
 if h.duplicated(list(ID)).any():raise RuntimeError('history candidate duplicate')
 return h
def _iso(h:pd.DataFrame)->IsotonicRegression:
 return IsotonicRegression(out_of_bounds='clip').fit(h.raw_score,h[NET])
def pava_strict(x:np.ndarray,y:np.ndarray,eps:float=1e-12)->np.ndarray:
 z=IsotonicRegression(out_of_bounds='clip').fit_transform(x,y); ranks=pd.Series(x).rank(method='first').to_numpy();return z+eps*ranks
def huber_tanh(h:pd.DataFrame,raw:np.ndarray)->np.ndarray:
 x=h.raw_score.to_numpy(float); y=h[NET].to_numpy(float); med=np.median(x);scale=np.median(np.abs(x-med)) or 1.;z=(x-med)/scale
 X=np.c_[np.ones(len(z)),np.tanh(z)];w=np.ones(len(z));beta=np.zeros(2)
 for _ in range(20):
  beta=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0];r=y-X@beta;s=max(np.median(np.abs(r))/0.6745,1e-6);w=np.minimum(1.,1.5*s/np.maximum(np.abs(r),1e-12))
 return beta[0]+beta[1]*np.tanh((raw-med)/scale)
def _pct_curve(h:pd.DataFrame,ev:pd.DataFrame)->np.ndarray:
 hp=h.groupby(TIME,sort=False).raw_score.rank(pct=True,method='average').to_numpy(float)
 q=np.linspace(.025,.975,20); knots=np.quantile(hp,q); means=[]
 for k in knots:
  take=np.argsort(np.abs(hp-k),kind='mergesort')[:max(200,len(h)//50)];means.append(float(h.iloc[take][NET].mean()))
 vals=pava_strict(knots,np.asarray(means)); ep=ev.groupby(TIME,sort=False).raw_score.rank(pct=True,method='first').to_numpy(float)
 return np.interp(ep,knots,vals,left=vals[0],right=vals[-1])
def timestamp_percentile(h:pd.DataFrame,ev:pd.DataFrame)->np.ndarray:
 # Side curves are shrunk to the pooled curve with the declared fixed support;
 # a final timestamp-local PAVA projection restores strict raw-score order.
 pooled=_pct_curve(h,ev);out=pooled.copy()
 for side in ('long','short'):
  bm=ev.side_name.eq(side).to_numpy();hs=h[h.side_name.eq(side)]
  if bm.any() and len(hs)>=SIDE:
   out[bm]=pooled[bm]+len(hs)/(len(hs)+LAMBDA)*(_pct_curve(hs,ev.loc[bm])-pooled[bm])
 for _ts,idx in ev.groupby(TIME,sort=False).groups.items():
  pos=ev.index.get_indexer(pd.Index(idx)); order_local=np.argsort(ev.loc[idx,'raw_score'].to_numpy(float),kind='mergesort')
  raw=ev.loc[idx,'raw_score'].to_numpy(float)[order_local];mapped=out[pos][order_local]
  repaired=pava_strict(raw,mapped);inverse=np.empty(len(pos),int);inverse[order_local]=np.arange(len(pos));out[pos]=repaired[inverse]
 return out
def map_day(h:pd.DataFrame,e:pd.DataFrame)->tuple[pd.DataFrame,dict]:
 out=e[list(ID)].copy();out['baseline']=np.nan;out['M1_strict_pooled']=np.nan;out['M2_huber_tanh']=np.nan;out['M3_timestamp_pct']=np.nan;out['eligible']=False
 snapshot=e[TIME].dt.floor('D').iloc[0]
 audit={'snapshot_utc':snapshot,'evaluation_rows':len(e),'reference_rows':len(h),'reference_min_end':h[END].min() if len(h) else pd.NaT,'reference_max_end':h[END].max() if len(h) else pd.NaT,'causal_label_window':bool(h[END].ge(snapshot-WINDOW).all() and h[END].lt(snapshot).all()),'eval_ref_id_overlap':len(set(e.candidate_id)&set(h.candidate_id)),'pooled_pass':len(h)>=POOL and h.raw_score.nunique()>1}
 if audit['eval_ref_id_overlap']:raise RuntimeError('evaluation/reference identity overlap')
 if not audit['pooled_pass']:return out,audit
 raw=e.raw_score.to_numpy(float);pooled=_iso(h);p=pooled.predict(raw);out['M1_strict_pooled']=pava_strict(raw,p);out['M2_huber_tanh']=huber_tanh(h,raw);out['M3_timestamp_pct']=timestamp_percentile(h,e);out['baseline']=p
 for side in ('long','short'):
  bm=e.side_name.eq(side).to_numpy();hs=h[h.side_name.eq(side)];audit[side+'_reference_rows']=len(hs)
  if len(hs)>=SIDE and hs.raw_score.nunique()>1:
   local=_iso(hs);w=len(hs)/(len(hs)+LAMBDA);out.loc[bm,'baseline']=p[bm]+w*(local.predict(raw[bm])-pooled.predict(raw[bm]));audit[side+'_weight']=w
  else:audit[side+'_weight']=0.
 out['eligible']=True;audit['coverage']=1.;return out,audit
def causal_maps(x:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
 parts=[];aud=[]
 for day,e in x.groupby(x[TIME].dt.floor('D'),sort=True):
  h=current_history(x,pd.Timestamp(day));m,a=map_day(h,e.copy());parts.append(m);aud.append(a)
 return x.merge(pd.concat(parts,ignore_index=True),on=list(ID),validate='one_to_one'),pd.DataFrame(aud)
def tie_metric(x:pd.DataFrame,col:str,f:float)->dict:
 q=order(x,col,f);n=len(q);cut=float(q[col].iloc[-1]);above=x[x[col]>cut];tie=x[np.isclose(x[col],cut,rtol=0,atol=1e-14)];need=n-len(above);expected=(above[NET].sum()+need*tie[NET].mean())/n
 return {'top_fraction':f,'selected_rows':n,'cutoff':cut,'tie_population':len(tie),'tie_selected_share':need/n,'expected_net_bps':float(expected*1e4),'gross_bps':float((above[GROSS].sum()+need*tie[GROSS].mean())/n*1e4),'cost_bps':float((above[COST].sum()+need*tie[COST].mean())/n*1e4)}
def metrics(x:pd.DataFrame,method:str,stage:str)->list[dict]:
 return [{'method':method,'stage':stage,**tie_metric(x,method,f),'candidate_rows':len(x),'rank_ic':float(x[method].corr(x[NET],method='spearman')),'global_topk_only':True} for f in TOPS]
def selection(economics:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for m in economics.method.unique():
  x=economics[(economics.method==m)&(economics.stage.str.startswith('march_fold_'))&(economics.top_fraction==.1)].sort_values('stage');v=x.expected_net_bps.to_numpy(float)
  rows.append({'method':m,'march_mean_bps':float(v.mean()),'march_std_bps':float(v.std()),'march_worst_bps':float(v.min()),'march_latest_bps':float(v[-1]),'objective':float(v.mean()-.5*v.std()+.25*v.min()),'folds':json.dumps(dict(zip(x.stage,v)))})
 return pd.DataFrame(rows).sort_values(['objective','method'],ascending=[False,True],kind='mergesort')
def run(source:Path=SOURCE,output_dir:Path=DEFAULT)->dict:
 if output_dir.exists():raise FileExistsError(output_dir)
 for p in (source/'manifest.json',source/'manifest.sha256',source/'scores.parquet'):
  if not p.is_file():raise FileNotFoundError(p)
 if sha(source/'manifest.json')!= (source/'manifest.sha256').read_text().split()[0]:raise RuntimeError('source manifest seal mismatch')
 sm=json.loads((source/'manifest.json').read_text());
 if sm.get('schema')!='canonical_execution_reliability_ablation_v2':raise RuntimeError('sealed v2 score ledger required')
 cols=[*ID,TIME,END,NET,GROSS,COST,'candidate_month','raw_score','score_available_utc','outer_fold','candidate_score_is_oof','config']
 x=pd.read_parquet(source/'scores.parquet',columns=cols);x=x[x.config.eq(CONFIG)].copy()
 for c in (TIME,END,'score_available_utc','__ts__'):x[c]=pd.to_datetime(x[c],utc=True)
 if len(x)!=94602 or x.duplicated(list(ID)).any() or not x[END].eq(x[TIME]+pd.Timedelta(hours=12)).all():raise RuntimeError('frozen score cohort contract')
 if not np.allclose(x[GROSS]-x[COST],x[NET],atol=1e-12):raise RuntimeError('exact cost contract')
 z,audit=causal_maps(x);z=z[z.eligible].copy();methods=['baseline','M1_strict_pooled','M2_huber_tanh','M3_timestamp_pct'];econ=[]
 march=z[z.candidate_month.eq('2025-03')]; april=z[z.candidate_month.eq('2025-04')]
 for m in methods:
  econ+=metrics(march,m,'march_aggregate')
  for f in ('selection_1','selection_2','selection_3'):econ+=metrics(march[march.outer_fold.eq(f)],m,'march_fold_'+f)
  econ+=metrics(april,m,'april_aggregate');econ+=metrics(april[april[TIME].ge(april[TIME].max().floor('D')-pd.Timedelta(days=6))],m,'april_latest7d')
 economics=pd.DataFrame(econ);sel=selection(economics);inv=[]
 for m in methods:
  for day,d in z.groupby(z[TIME].dt.floor('D')):
   rr=d.raw_score.rank(method='first',ascending=False);mr=d[m].rank(method='first',ascending=False);inv.append({'method':m,'snapshot_utc':day,'inversion_rate':float(np.mean(np.sign(rr.to_numpy()[:,None]-rr.to_numpy()[None,:])!=np.sign(mr.to_numpy()[:,None]-mr.to_numpy()[None,:]))),'raw_unique':d.raw_score.nunique(),'mapped_unique':d[m].nunique()})
 attr=[]; transport=[]
 for m in methods:
  q=order(april,m,.1)
  for side,d in q.groupby('side_name'):attr.append({'method':m,'dimension':'side','value':side,'rows':len(d),'share':len(d)/len(q),'net_bps':float(d[NET].mean()*1e4)})
  for asset,d in q.groupby('__symbol__'):attr.append({'method':m,'dimension':'asset','value':asset,'rows':len(d),'share':len(d)/len(q),'net_bps':float(d[NET].mean()*1e4)})
  for stage,part in (('march',march),('april',april)):
   raw_book=order(part,'raw_score',.1); map_book=order(part,m,.1)
   transport.append({'method':m,'stage':stage,'metric':'raw_map_global_rank_correlation','value':float(part.raw_score.corr(part[m],method='spearman'))})
   transport.append({'method':m,'stage':stage,'metric':'raw_map_global_top10_overlap','value':float(len(set(raw_book.candidate_id)&set(map_book.candidate_id))/len(map_book))})
   transport.append({'method':m,'stage':stage,'metric':'raw_score_iqr','value':float(part.raw_score.quantile(.75)-part.raw_score.quantile(.25))})
   transport.append({'method':m,'stage':stage,'metric':'mapped_score_iqr','value':float(part[m].quantile(.75)-part[m].quantile(.25))})
  previous=None
  for day,d in april.groupby(april[TIME].dt.floor('D'),sort=True):
   book=order(d,m,.1);symbols=set((book.side_name.astype(str)+'|'+book.__symbol__.astype(str)))
   if previous is not None: transport.append({'method':m,'stage':'april','snapshot_utc':day,'metric':'day_to_day_symbol_side_top10_turnover','value':float(1-len(symbols&previous)/max(1,len(symbols|previous)))})
   previous=symbols
 gates=[]
 for m in methods:
  a10=economics[(economics.method==m)&(economics.stage=='april_aggregate')&(economics.top_fraction==.1)].iloc[0];w10=economics[(economics.method==m)&(economics.stage=='april_latest7d')&(economics.top_fraction==.1)].iloc[0];s=pd.DataFrame(attr);s=s[(s.method==m)&(s.dimension=='side')]
  gates += [{'method':m,'gate':'April global top10 expected net positive','pass':a10.expected_net_bps>0,'detail':a10.expected_net_bps},{'method':m,'gate':'April latest7d top10 positive','pass':w10.expected_net_bps>0,'detail':w10.expected_net_bps},{'method':m,'gate':'April top10 tie selected share <=5%','pass':a10.tie_selected_share<=.05,'detail':a10.tie_selected_share},{'method':m,'gate':'both side contributions positive','pass':(s.net_bps>0).all(),'detail':json.dumps(dict(zip(s.value,s.net_bps)))}]
 gates=pd.DataFrame(gates);stage=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'.',dir=output_dir.parent))
 try:
  z.to_parquet(stage/'mapped_scores.parquet',index=False,compression='zstd');audit.to_csv(stage/'causal_mapping_audit.csv',index=False);economics.to_csv(stage/'economics.csv',index=False);sel.to_csv(stage/'march_selection.csv',index=False);pd.DataFrame(inv).to_csv(stage/'order_inversion_audit.csv',index=False);pd.DataFrame(attr).to_csv(stage/'april_top10_attribution.csv',index=False);pd.DataFrame(transport).to_csv(stage/'transport_and_turnover.csv',index=False);gates.to_csv(stage/'promotion_gates.csv',index=False)
  hashes={p.name:sha(p) for p in stage.iterdir() if p.is_file()};manifest={'schema':'canonical_execution_reliability_mapping_ablation_v1','status':'SEALED_RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY','promotion_eligible':False,'run_id':'canonical_execution_reliability_mapping_ablation_20260730_v1','source':{'artifact':str(source),'scores_sha256':sha(source/'scores.parquet'),'manifest_sha256':sha(source/'manifest.json'),'config':CONFIG},'contract':{'dataset':'sealed canonical execution-reliability v2 scores only','labels':'exact deployed-policy net EV H12','geometry':'no predictive retraining/HPO; fixed 21d causal mapping constants','universe':'sealed v2 candidate cohort, global cross-side/cross-timestamp book','costs':'one execution_cost_return; gross-cost=net','train_eval_utc':'causal mapping refs S-21d <= label_end < S; March OOF selection, April frozen diagnostic','purge_embargo':'H12 labels resolved strictly before map snapshot; no evaluation/reference candidate overlap','models':'baseline isotonic+shrunk side; M1 strict pooled isotonic; M2 fixed Huber/tanh; M3 fixed 20-knot timestamp percentile PAVA','policy':'research-only global topK, no side quotas, no replay','random_seeds':'not applicable (deterministic maps)','code_revision':'not available in source manifest; runner hash sealed below'},'constants':{'window_days':21,'pooled_min':POOL,'side_min':SIDE,'side_lambda':LAMBDA,'m3_knots':20},'outputs_sha256':hashes,'runner_sha256':sha(Path(__file__))};write(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(sha(stage/'manifest.json')+'  manifest.json\n');write(stage/'seal.json',{'status':manifest['status'],'manifest_sha256':sha(stage/'manifest.json'),'files_sha256':{p.name:sha(p) for p in stage.iterdir() if p.is_file()}});os.replace(stage,output_dir)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output-dir',type=Path,default=DEFAULT);a=p.parse_args();print(json.dumps(run(a.source,a.output_dir),indent=2))
