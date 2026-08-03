#!/usr/bin/env python3
"""Causal opportunity-hurdle and rank-preserving mapping ablations.

This research runner uses exact-policy outcomes only for rows whose labels have
resolved before the evaluation month.  It never creates side/time/state quotas:
each arm fills one pooled global 10% book from the same held-out candidates.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data_perp/artifacts'; OUT=ART/'causal_opportunity_hurdle_mapping_ablation_20260730_v3'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def safe(x:Any)->Any:
 if isinstance(x,(Path,pd.Timestamp)):return str(x)
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def top10(d:pd.DataFrame,score:str)->pd.Series:
 n=max(1,math.ceil(len(d)*.1));out=pd.Series(False,index=d.index)
 out.loc[d.sort_values([score,'candidate_id'],ascending=[False,True],kind='stable').index[:n]]=True;return out
def quantile_rank(value:pd.Series,ref:pd.Series)->pd.Series:
 # Prior/training empirical coordinate; ties have a deterministic midpoint.
 s=np.sort(ref.dropna().to_numpy(float));return value.apply(lambda x: float(np.searchsorted(s,x,side='right')/len(s)) if len(s) and np.isfinite(x) else np.nan)
def resolved_before(d:pd.DataFrame,evaluation_start:pd.Timestamp)->pd.DataFrame:
 """Keep only candidates whose exact 12h outcome is available at the cutoff."""
 return d.loc[d['__ts__']+pd.Timedelta(hours=12)<evaluation_start].copy()
def load(art:Path)->pd.DataFrame:
 h=pd.read_parquet(art/'historical_causal_score_economics_mapping_20260729_v1/canonical_residual__score_residual_expected_ev/causal_mapped_candidates.parquet')
 h=pd.DataFrame({'candidate_id':h.candidate_id,'__ts__':h['__ts__'],'side_name':h.side_name,'lineage':'canonical_2025','raw':h.score_raw,'mapped':h.mapped_direct_net,'refs':h.map_reference_rows,'net':h.execution_net_ev_12h,'gross':h.execution_gross_ev_12h,'cost':h.execution_cost_return,'opportunity':h.opportunity_gross_above_cost_0bps}).loc[h.mapped_eligible].copy()
 c=pd.read_parquet(art/'current_exact_policy_global_book_mapping_source_20260730_v2/causal_mapped_candidates.parquet')
 c=pd.DataFrame({'candidate_id':c.candidate_id,'__ts__':c['__ts__'],'side_name':c.side_name,'lineage':'current_2026','raw':c['catboost__residual__without_hpo__all_features'],'mapped':c.mapped_direct_net,'refs':c.map_reference_rows,'net':c.execution_net_ev_12h,'gross':c.execution_gross_ev_12h,'cost':c.execution_cost_return,'opportunity':c.opportunity_gross_above_cost_0bps}).loc[c.mapped_eligible].copy()
 d=pd.concat([h,c],ignore_index=True);d['__ts__']=pd.to_datetime(d['__ts__'],utc=True);d['month']=d['__ts__'].dt.strftime('%Y-%m');d['week']=d['__ts__'].dt.to_period('W-SUN').astype(str);d.opportunity=d.opportunity.astype(int);return d
def features(train:pd.DataFrame,test:pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
 # Raw score, side and its causal rank are all decision-time fields.  Mapping
 # output is intentionally excluded from the *pre-mapping* opportunity hurdle.
 mu=train.raw.mean();sd=max(train.raw.std(),1e-8)
 def x(d):return np.column_stack([(d.raw-mu)/sd,d.side_name.eq('long').astype(float)])
 return x(train),x(test)
def arm_metrics(d:pd.DataFrame,selected:pd.Series,arm:str,lineage:str,month:str,params:dict[str,Any])->dict[str,Any]:
 g=d.loc[selected];daily=g.groupby(g['__ts__'].dt.floor('D')).net.mean()
 return {'lineage':lineage,'evaluation_month':month,'arm':arm,'population_rows':len(d),'selected_rows':len(g),'selected_days':daily.size,'mean_net_bps':g.net.mean()*1e4,'mean_gross_bps':g.gross.mean()*1e4,'mean_cost_bps':g.cost.mean()*1e4,'opportunity_rate':g.opportunity.mean(),'net_day_q10_bps':daily.quantile(.1)*1e4,'net_day_q50_bps':daily.quantile(.5)*1e4,'net_day_q90_bps':daily.quantile(.9)*1e4,'long_share':g.side_name.eq('long').mean(),**params}
def selected_attribution(d:pd.DataFrame,selected:pd.Series,arm:str,lineage:str,month:str)->pd.DataFrame:
 """Retain membership chosen from the monthly global book for later attribution.

 This function does not re-rank by week: the weekly table is only a view of the
 already-frozen monthly selection.
 """
 g=d.loc[selected,['__ts__','week','net','gross','cost','opportunity','side_name']].copy()
 return g.assign(arm=arm,lineage=lineage,evaluation_month=month)
def choose_hurdle(train:pd.DataFrame,model:LogisticRegression)->tuple[float,float]:
 x,_=features(train,train);p=model.predict_proba(x)[:,1];best=(0.0,-np.inf)
 for t in (.20,.30,.40,.50,.60,.70,.80):
  eligible=p>=t
  if eligible.mean()<.10:continue
  tmp=train.loc[eligible].copy();score=tmp.mapped;metric=tmp.loc[top10(tmp,'mapped'),'net'].mean()
  if metric>best[1]:best=(t,float(metric))
 return best
def choose_mapping(train:pd.DataFrame)->tuple[float,float,float]:
 best=(1.0,0.0,-np.inf)
 for lam in (0,.25,.5,.75,1):
  for tau in (0,500,2000,10000):
   raw=quantile_rank(train.raw,train.raw);mapped=quantile_rank(train.mapped,train.mapped);w=train.refs/(train.refs+tau) if tau else pd.Series(1.,index=train.index)
   score=(1-lam*w)*raw+(lam*w)*mapped;tmp=train.assign(_score=score);metric=tmp.loc[top10(tmp,'_score'),'net'].mean()
   if metric>best[2]:best=(lam,tau,float(metric))
 return best
def run(*,artifacts:Path=ART,output:Path=OUT)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 d=load(artifacts);reports=[];choices=[];attribution=[]
 for lineage,all_rows in d.groupby('lineage',sort=True):
  months=sorted(all_rows.month.unique())
  for i,month in enumerate(months):
   if i==0:continue
   evaluation_start=pd.Timestamp(month+'-01',tz='UTC')
   # The outcome is an exact 12h policy outcome, so strip any late prior-month
   # row whose label would not yet have resolved at the held-out month boundary.
   train=resolved_before(all_rows.loc[all_rows.month.isin(months[:i])],evaluation_start);test=all_rows.loc[all_rows.month.eq(month)].copy()
   # No held-out outcome enters either model fit or HPO choice.
   x_train,x_test=features(train,test);model=LogisticRegression(C=1.0,max_iter=500,random_state=20260730).fit(x_train,train.opportunity)
   threshold,train_hurdle_net=choose_hurdle(train,model);p=model.predict_proba(x_test)[:,1]
   baseline=top10(test,'mapped');reports.append(arm_metrics(test,baseline,'baseline_causal_map',lineage,month,{}));attribution.append(selected_attribution(test,baseline,'baseline_causal_map',lineage,month))
   admitted=pd.Series(p>=threshold,index=test.index)
   if admitted.sum()>=math.ceil(len(test)*.1):
    score=test.mapped.where(admitted,-np.inf);selected=top10(test.assign(_score=score),'_score');reports.append(arm_metrics(test,selected,'opportunity_hurdle_before_mapping',lineage,month,{'hurdle_threshold':threshold,'train_hurdle_top10_net_bps':train_hurdle_net*1e4,'admission_rate':admitted.mean()}));attribution.append(selected_attribution(test,selected,'opportunity_hurdle_before_mapping',lineage,month))
   lam,tau,train_map_net=choose_mapping(train);raw=quantile_rank(test.raw,train.raw);mapped=quantile_rank(test.mapped,train.mapped);w=test.refs/(test.refs+tau) if tau else pd.Series(1.,index=test.index);score=(1-lam*w)*raw+(lam*w)*mapped;selected=top10(test.assign(_score=score),'_score')
   reports.append(arm_metrics(test,selected,'rank_preservation_support_shrinkage',lineage,month,{'lambda_mapped':lam,'support_tau':tau,'train_mapping_top10_net_bps':train_map_net*1e4}));attribution.append(selected_attribution(test,selected,'rank_preservation_support_shrinkage',lineage,month))
   choices.append({'lineage':lineage,'evaluation_month':month,'training_months':'|'.join(months[:i]),'hurdle_threshold':threshold,'mapping_lambda':lam,'support_tau':tau,'training_rows':len(train),'heldout_rows':len(test)})
 results=pd.DataFrame(reports);choices=pd.DataFrame(choices);selected_rows=pd.concat(attribution,ignore_index=True)
 # Gate deliberately demands robust daily lower tail at both frequencies and
 # both the aggregate and latest evaluation periods.
 weekly=[]
 for keys,g in selected_rows.groupby(['arm','lineage','evaluation_month','week'],sort=True):
  daily=g.groupby(g['__ts__'].dt.floor('D')).net.mean()
  weekly.append({'arm':keys[0],'lineage':keys[1],'evaluation_month':keys[2],'week':keys[3],'selected_rows':len(g),'selected_days':daily.size,'mean_net_bps':g.net.mean()*1e4,'mean_gross_bps':g.gross.mean()*1e4,'mean_cost_bps':g.cost.mean()*1e4,'opportunity_rate':g.opportunity.mean(),'net_day_q10_bps':daily.quantile(.1)*1e4,'net_day_q50_bps':daily.quantile(.5)*1e4,'net_day_q90_bps':daily.quantile(.9)*1e4,'long_share':g.side_name.eq('long').mean()})
 weekly=pd.DataFrame(weekly)
 gates=[]
 for frequency,frame,period in [('monthly',results,'evaluation_month'),('weekly',weekly,'week')]:
  for arm,g in frame.groupby('arm'):
   latest=g.loc[g[period].eq(g[period].max())];agg=g
   for scope,x in [('aggregate',agg),('latest',latest)]:
    gates.append({'arm':arm,'frequency':frequency,'scope':scope,'periods':x[period].nunique(),'mean_net_bps':x.mean_net_bps.mean(),'q10_of_period_daily_q10_bps':x.net_day_q10_bps.quantile(.1),'q50_of_period_daily_q50_bps':x.net_day_q50_bps.quantile(.5),'passes_frozen_arm_gate':bool((x.mean_net_bps>0).all() and (x.net_day_q10_bps>0).all() and (x.net_day_q50_bps>0).all())})
 gates=pd.DataFrame(gates)
 # An arm must clear every aggregate/latest and weekly/monthly gate.  This is
 # deliberately stricter than selecting whichever view happened to look best.
 frozen_arms=gates.groupby('arm').passes_frozen_arm_gate.all();frozen=bool(frozen_arms.any())
 output.mkdir(parents=True);results.to_csv(output/'heldout_monthly_arm_metrics.csv',index=False);weekly.to_csv(output/'heldout_weekly_attribution.csv',index=False);choices.to_csv(output/'causal_hpo_choices.csv',index=False);gates.to_csv(output/'weekly_monthly_q10_q50_gates.csv',index=False)
 pd.DataFrame([{'portfolio_replay_run':False,'reason':'no arm passes frozen latest-and-aggregate Q10/Q50 gate'}]).to_csv(output/'portfolio_replay_status.csv',index=False)
 inputs={'historical':artifacts/'historical_causal_score_economics_mapping_20260729_v1/canonical_residual__score_residual_expected_ev/manifest.json','current':artifacts/'current_exact_policy_global_book_mapping_source_20260730_v2/manifest.json'}
 manifest={'schema':'causal_opportunity_hurdle_mapping_ablation_v3','research_only':True,'promotion_eligible':False,'selection_contract':'one pooled global top10 per held-out month; no side/timestamp/regime/phase quotas; weekly attribution preserves that monthly selection and never re-ranks','hpo_contract':'threshold/lambda/tau chosen only on earlier candidates whose exact 12h outcome resolved strictly before the held-out calendar-month boundary, within lineage','portfolio_replay':False,'frozen_arm_passed':frozen,'counts':{'candidate_rows':len(d),'evaluated_rows':len(results),'evaluated_months':sorted(results.evaluation_month.unique()),'evaluated_weeks':sorted(weekly.week.unique())},'inputs_sha256':{k:sha(v) for k,v in inputs.items()},'outputs_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
 p=output/'manifest.json';p.write_text(json.dumps(safe(manifest),indent=2,sort_keys=True)+'\n');(output/'manifest.sha256').write_text(f"{sha(p)}  manifest.json\n");return manifest
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,default=ART);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args(argv);print(json.dumps(safe(run(artifacts=a.artifacts,output=a.output)),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
