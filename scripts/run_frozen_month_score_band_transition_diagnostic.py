#!/usr/bin/env python3
"""Frozen monthly base-score band transitions on identical canonical rows.

Numeric pooled-global source-month cutoffs are frozen, then applied unchanged
to the next month.  Target-local bins are a comparator only.  No candidate IDs
are matched across months and no timestamp/side re-selection is performed.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet'
DEFAULT_OUTPUT=ROOT/'data_perp/artifacts/frozen_month_score_band_transition_diagnostic_20260730_v2'
PAIRS=(('2025-02','2025-03'),('2025-03','2025-04'))
SPECS={'ventile':20,'decile':10}; TOPS=(.01,.05,.10,.20); BOOT=50; MIN_ROWS=30; MIN_DAYS=3; SEED=20260730

def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def safe(x:Any)->Any:
 if x is pd.NaT or (not isinstance(x,(Mapping,list,tuple)) and pd.isna(x)): return None
 if isinstance(x,(Path,pd.Timestamp)):return str(x)
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Mapping):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x
def write_json(p:Path,x:Mapping[str,Any])->None:
 t=p.with_name(f'.{p.name}.{os.getpid()}.tmp');t.write_text(json.dumps(safe(dict(x)),indent=2,sort_keys=True)+'\n');os.replace(t,p)
def corr(a:pd.Series,b:pd.Series)->float:
 q=pd.DataFrame({'a':a,'b':b}).dropna()
 if len(q)<3 or q.a.nunique()<2 or q.b.nunique()<2:return np.nan
 x=spearmanr(q.a,q.b).statistic;return float(x) if np.isfinite(x) else np.nan
def thresholds(source:pd.Series,bands:int)->np.ndarray:
 x=pd.to_numeric(source,errors='raise').to_numpy(float)
 if len(x)<MIN_ROWS or np.unique(x).size<bands: raise ValueError('insufficient unique source score support for frozen bands')
 return np.quantile(x,np.arange(1,bands)/bands,method='linear')
def assign(values:pd.Series,cuts:np.ndarray)->np.ndarray:
 return np.searchsorted(cuts,pd.to_numeric(values,errors='raise').to_numpy(float),side='right').astype(int)
def cutoff_audit(source:pd.Series,cuts:np.ndarray,kind:str,pair:str,evaluation_month:str,role:str)->pd.DataFrame:
 v=source.to_numpy(float);rows=[]
 for i,c in enumerate(cuts,1):
  plateau=v==c;rows.append({'pair':pair,'kind':kind,'evaluation_month':evaluation_month,'threshold_role':role,'boundary':i,'cutoff':float(c),'rows_above':int((v>c).sum()),'plateau_rows':int(plateau.sum()),'rows_at_or_above':int((v>=c).sum()),'tie_ambiguous':bool(plateau.sum()>1)})
 return pd.DataFrame(rows)
def bootstrap(frame:pd.DataFrame,seed:int)->dict[str,float]:
 days=frame['_day'].drop_duplicates().to_numpy()
 if len(frame)<MIN_ROWS or len(days)<MIN_DAYS:return {'block_supported':False,'net_mean_p025_bps':np.nan,'net_mean_p975_bps':np.nan,'net_ic_p025':np.nan,'net_ic_p975':np.nan}
 rng=np.random.default_rng(seed); nets=[];ics=[]
 by={d:g for d,g in frame.groupby('_day',sort=False)}
 for _ in range(BOOT):
  sample=pd.concat([by[d] for d in rng.choice(days,len(days),replace=True)],ignore_index=True)
  nets.append(float(sample.execution_net_ev_12h.mean()*1e4));ics.append(corr(sample.base_oof_score,sample.execution_net_ev_12h))
 return {'block_supported':True,'net_mean_p025_bps':float(np.nanquantile(nets,.025)),'net_mean_p975_bps':float(np.nanquantile(nets,.975)),'net_ic_p025':float(np.nanquantile(ics,.025)),'net_ic_p975':float(np.nanquantile(ics,.975))}
def band_metrics(frame:pd.DataFrame,base:dict[str,Any],seed:int)->dict[str,Any]:
 n=len(frame);opp=frame.opportunity_gross_above_cost_0bps.astype(bool);positive=frame.execution_net_ev_12h.gt(0);adverse=~positive
 out={**base,'rows':n,'days':int(frame._day.nunique()),'score_mean':float(frame.base_oof_score.mean()),'native_target_rank_ic':corr(frame.base_oof_score,frame.__first_touch_target_soft__),'exact_gross_rank_ic':corr(frame.base_oof_score,frame.execution_gross_ev_12h),'exact_net_rank_ic':corr(frame.base_oof_score,frame.execution_net_ev_12h),'opportunity_incidence':float(opp.mean()),'opportunity_precision_net_positive':float(positive.loc[opp].mean()) if opp.any() else np.nan,'favorable_gross_mean_bps':float(frame.loc[opp,'execution_gross_ev_12h'].mean()*1e4) if opp.any() else np.nan,'favorable_net_mean_bps':float(frame.loc[positive,'execution_net_ev_12h'].mean()*1e4) if positive.any() else np.nan,'adverse_net_mean_bps':float(frame.loc[adverse,'execution_net_ev_12h'].mean()*1e4) if adverse.any() else np.nan,'mfe_mean_bps':float(frame.execution_mfe_return_12h.mean()*1e4),'mae_mean_bps':float(frame.execution_mae_return_12h.mean()*1e4),'cost_mean_bps':float(frame.execution_cost_return.mean()*1e4),'gross_mean_bps':float(frame.execution_gross_ev_12h.mean()*1e4),'net_mean_bps':float(frame.execution_net_ev_12h.mean()*1e4),'full_stop_rate':float(frame.execution_exit_class.eq('full_stop').mean()),'timeout_rate':float(frame.execution_exit_class.eq('timeout').mean()),'trailing_rate':float(frame.execution_exit_class.eq('trailing').mean()),'adverse_exit_rate':float(frame.execution_exit_class.eq('adverse_exit').mean())}
 # Side rows are attribution only.  The expensive UTC-day uncertainty is
 # deliberately computed once for each pooled-global band, never treated as
 # a side-local selection statistic.
 out.update(bootstrap(frame,seed) if base['scope']=='pooled_global' else {'block_supported':False,'net_mean_p025_bps':np.nan,'net_mean_p975_bps':np.nan,'net_ic_p025':np.nan,'net_ic_p975':np.nan});return out
def expected_top_coverage()->set[tuple[str,str,str,float]]:
 return {(f'{source}->{target}',month,scheme,depth) for source,target in PAIRS for month,scheme in ((source,'source_frozen'),(target,'source_frozen'),(target,'target_local')) for depth in TOPS}
def validate_top_coverage(top:pd.DataFrame)->None:
 found=set(tuple(x) for x in top.loc[:,['pair','evaluation_month','scheme','top_fraction']].drop_duplicates().itertuples(index=False,name=None))
 if found!=expected_top_coverage():raise ValueError(f'incomplete fixed/top coverage: missing={expected_top_coverage()-found}, extra={found-expected_top_coverage()}')
def top_contrib(frame:pd.DataFrame,score:str,fraction:float,band:str,pair:str,month:str,scheme:str,cutoff:float|None=None)->list[dict[str,Any]]:
 cut=float(np.quantile(frame[score].to_numpy(float),1-fraction,method='linear')) if cutoff is None else float(cutoff);sel=frame.loc[frame[score].ge(cut)].copy(); total=float(sel.execution_net_ev_12h.sum())
 rows=[]
 for b,g in sel.groupby(band,sort=True): rows.append({'pair':pair,'evaluation_month':month,'scheme':scheme,'top_fraction':fraction,'selection_cutoff':cut,'selected_rows':len(sel),'selection_rate':len(sel)/len(frame),'cutoff_plateau_rows':int(frame[score].eq(cut).sum()),'band':int(b),'band_selected_rows':len(g),'band_selected_share':len(g)/len(sel),'band_net_contribution_bps_sum':float(g.execution_net_ev_12h.sum()*1e4),'band_net_contribution_share':float(g.execution_net_ev_12h.sum()/total) if total else np.nan,'band_selected_mean_net_bps':float(g.execution_net_ev_12h.mean()*1e4)})
 return rows
def run(output_dir:Path)->dict[str,Any]:
 if output_dir.exists():raise FileExistsError(f'immutable output exists: {output_dir}')
 x=pd.read_parquet(PANEL,columns=['candidate_id','candidate_month','__ts__','side_name','__symbol__','base_oof_score','__first_touch_target_soft__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','execution_mfe_return_12h','execution_mae_return_12h','opportunity_gross_above_cost_0bps','execution_exit_class'])
 x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['_day']=x.__ts__.dt.floor('D')
 req={'2025-02','2025-03','2025-04'}
 if set(x.candidate_month.unique())!=req or x.candidate_id.duplicated().any():raise ValueError('canonical identical-row contract fails')
 all_bands=[];all_cuts=[];all_migration=[];all_top=[]
 for source_month,target_month in PAIRS:
  source=x.loc[x.candidate_month.eq(source_month)].copy();target=x.loc[x.candidate_month.eq(target_month)].copy();pair=f'{source_month}->{target_month}'
  for kind,n in SPECS.items():
   cuts=thresholds(source.base_oof_score,n);all_cuts.extend([cutoff_audit(source.base_oof_score,cuts,kind,pair,source_month,'source_threshold_definition'),cutoff_audit(target.base_oof_score,cuts,kind,pair,target_month,'target_application_of_frozen_source_threshold')])
   source[f'frozen_{kind}']=assign(source.base_oof_score,cuts);target[f'frozen_{kind}']=assign(target.base_oof_score,cuts)
   target_cuts=thresholds(target.base_oof_score,n);target[f'local_{kind}']=assign(target.base_oof_score,target_cuts)
   for month,frame,scheme,col in [(source_month,source,'source_frozen',f'frozen_{kind}'),(target_month,target,'source_frozen',f'frozen_{kind}'),(target_month,target,'target_local',f'local_{kind}')]:
    for scope,part in [('pooled_global',frame),*[(f'side_attribution::{s}',g) for s,g in frame.groupby('side_name',sort=True)]]:
     for b,g in part.groupby(col,sort=True):all_bands.append(band_metrics(g,{'pair':pair,'evaluation_month':month,'band_kind':kind,'scheme':scheme,'scope':scope,'band':int(b)},SEED+len(all_bands)))
    # A frozen top threshold is the source-month numeric cutoff.  Target-local
    # top thresholds are separately reported only as a scale-free comparator.
    if scheme=='source_frozen':
     all_top.extend(sum([top_contrib(frame,'base_oof_score',z,col,pair,month,scheme,float(np.quantile(source.base_oof_score,1-z,method='linear'))) for z in TOPS],[]))
    else:
     all_top.extend(sum([top_contrib(frame,'base_oof_score',z,col,pair,month,scheme) for z in TOPS],[]))
   # Migration is intentionally pooled only: same numeric source band, no ID matching.
   for b in range(n):
    a=source.loc[source[f'frozen_{kind}'].eq(b)];q=target.loc[target[f'frozen_{kind}'].eq(b)]
    all_migration.append({'pair':pair,'band_kind':kind,'band':b,'source_rows':len(a),'target_rows':len(q),'source_mass':len(a)/len(source),'target_mass':len(q)/len(target),'mass_delta':len(q)/len(target)-len(a)/len(source),'source_score_mean':float(a.base_oof_score.mean()),'target_score_mean':float(q.base_oof_score.mean()),'within_band_net_mean_delta_bps':float((q.execution_net_ev_12h.mean()-a.execution_net_ev_12h.mean())*1e4),'within_band_gross_ic_delta':corr(q.base_oof_score,q.execution_gross_ev_12h)-corr(a.base_oof_score,a.execution_gross_ev_12h),'within_band_net_ic_delta':corr(q.base_oof_score,q.execution_net_ev_12h)-corr(a.base_oof_score,a.execution_net_ev_12h),'interpretation':'mass/composition migration is separate from response deltas; no cross-month candidate matching'})
 stage=Path(tempfile.mkdtemp(dir=output_dir.parent,prefix=f'.{output_dir.name}.'))
 try:
  top=pd.DataFrame(all_top);validate_top_coverage(top)
  outputs={'frozen_cutoffs_tie_audit.csv':pd.concat(all_cuts,ignore_index=True),'band_metrics.csv':pd.DataFrame(all_bands),'band_mass_and_response_migration.csv':pd.DataFrame(all_migration),'fixed_band_global_top_contribution.csv':top}
  for n,t in outputs.items():t.to_csv(stage/n,index=False)
  manifest={'schema':'frozen_month_score_band_transition_diagnostic_v1','status':'DIAGNOSTIC_COMPLETE_NO_POLICY_CHANGE','promotion_eligible':False,'input':{'path':str(PANEL),'sha256':sha256(PANEL),'identical_canonical_rows':int(len(x))},'pairs':[f'{a}->{b}' for a,b in PAIRS],'contracts':{'cutoffs':'pooled-global numeric score ventile/decile cutoffs computed from source month labels-free score values only, then frozen unchanged for target month','local_comparator':'target-local pooled-global numeric bins are a descriptive comparator; never used to define source migration','selection':'no candidate matching across months; no timestamp/side re-selection; side is attribution only','economics':'exact canonical 12h gross/cost/net and deployed exit classes; global top thresholds report all cutoff-plateau rows and no deterministic tie-break','uncertainty':f'UTC-day block bootstrap {BOOT} replicates with 2.5/97.5 intervals; metrics fail closed below {MIN_ROWS} rows or {MIN_DAYS} days'},'outputs_sha256':{n:sha256(stage/n) for n in outputs},'runner':{'path':str(Path(__file__).resolve()),'sha256':sha256(Path(__file__).resolve())}}
  write_json(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(sha256(stage/'manifest.json')+'\n');os.replace(stage,output_dir)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();print(json.dumps(safe(run(a.output_dir)),sort_keys=True))
