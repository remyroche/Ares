#!/usr/bin/env python3
"""Explain alpha-IC versus exact execution-EV divergence without changing policy.

All tail slices use a single pooled global top 10% *within the reporting month*
over sides and timestamps.  State and transition phase are post-hoc descriptive
dimensions only; neither supplies a quota nor changes selection.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'data_perp/artifacts'
OUT=ART/'alpha_execution_ev_gap_diagnostic_20260730_v1'

def _sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as h:
  for b in iter(lambda:h.read(1<<20),b''):d.update(b)
 return d.hexdigest()

def _safe(x:Any)->Any:
 if isinstance(x,(Path,pd.Timestamp)):return str(x)
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):_safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [_safe(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def _json(p:Path,x:dict[str,Any])->None:
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(_safe(x),indent=2,sort_keys=True)+'\n');os.replace(t,p)

def global_topk(frame:pd.DataFrame, *, score:str, group:Sequence[str]=('lineage','month'), fraction:float=.10)->pd.Series:
 """Return global, cross-side/cross-timestamp tail membership with deterministic ties."""
 out=pd.Series(False,index=frame.index); valid=frame.dropna(subset=[score])
 groups=[(None,valid)] if not group else valid.groupby(list(group),sort=False)
 for _,g in groups:
  n=max(1,int(math.ceil(len(g)*fraction)))
  take=g.sort_values([score,'candidate_id'],ascending=[False,True],kind='stable').index[:n]
  out.loc[take]=True
 return out

def _daily_ci(g:pd.DataFrame)->tuple[float,float]:
 d=g.groupby(g['__ts__'].dt.floor('D'))['net'].mean().dropna()
 if len(d)<2:return (np.nan,np.nan)
 se=float(d.std(ddof=1)/np.sqrt(len(d)))
 return float(d.mean()-1.96*se),float(d.mean()+1.96*se)

def _summary(data:pd.DataFrame, cols:Sequence[str], label:str)->pd.DataFrame:
 rows=[]
 for key,g in data.groupby(list(cols),dropna=False,sort=True):
  key=(key,) if not isinstance(key,tuple) else key
  lo,hi=_daily_ci(g)
  alpha=g.dropna(subset=['alpha_score','alpha_target'])
  rows.append({"slice":label,**dict(zip(cols,key)),"selected_rows":len(g),"selected_days":g['__ts__'].dt.floor('D').nunique(),
   "mean_net_bps":g.net.mean()*1e4,"mean_gross_bps":g.gross.mean()*1e4,"mean_cost_bps":g.cost.mean()*1e4,
   "net_bps_ci95_low":lo*1e4,"net_bps_ci95_high":hi*1e4,"positive_net_rate":g.net.gt(0).mean(),
   "opportunity_rate":g.opportunity.mean(),"long_share":g.side_name.eq('long').mean(),
   "alpha_rank_ic":alpha.alpha_score.corr(alpha.alpha_target,method='spearman') if len(alpha)>=8 else np.nan,
   "mapping_reference_rows_median":g.map_reference_rows.median()})
 return pd.DataFrame(rows)

def _read_sources(art:Path)->pd.DataFrame:
 rec=pd.read_parquet(art/'reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet')
 a=pd.DataFrame({'candidate_id':rec.candidate_id,'__ts__':rec['__ts__'],'side_name':rec.side_name,'lineage':rec.stack_lineage,
  'selection_score':rec.score_residual_expected_ev,'raw_score':rec.score_residual_expected_ev,'mapped_score':np.nan,
  'alpha_score':rec.score_residual_alpha,'alpha_target':rec['__reconstructed_soft_alpha_12h__'],'gross':rec.execution_gross_ev_12h,'cost':rec.execution_cost_return,'net':rec.execution_net_ev_12h,
  'exit_reason':pd.NA,'opportunity':rec.execution_net_ev_12h.gt(0),'map_reference_rows':np.nan,'mapping_kind':'direct_model_ev_no_separate_causal_map'})
 hist=pd.read_parquet(art/'historical_causal_score_economics_mapping_20260729_v1/canonical_residual__score_residual_expected_ev/causal_mapped_candidates.parquet')
 b=pd.DataFrame({'candidate_id':hist.candidate_id,'__ts__':hist['__ts__'],'side_name':hist.side_name,'lineage':'canonical_residual_2025',
  'selection_score':hist.mapped_direct_net,'raw_score':hist.score_raw,'mapped_score':hist.mapped_direct_net,'alpha_score':hist.score_base_alpha,'alpha_target':hist['__first_touch_target_soft__'],
  'gross':hist.execution_gross_ev_12h,'cost':hist.execution_cost_return,'net':hist.execution_net_ev_12h,'exit_reason':hist.execution_exit_reason,
  'opportunity':hist.opportunity_gross_above_cost_0bps.astype(bool),'map_reference_rows':hist.map_reference_rows,'mapping_kind':'causal_21d_isotonic'}).loc[hist.mapped_eligible].copy()
 cur=pd.read_parquet(art/'current_exact_policy_global_book_mapping_source_20260730_v2/causal_mapped_candidates.parquet')
 ctx=pd.read_parquet(art/'execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet',columns=['candidate_id','base_oof_score','execution_exit_reason'])
 cur=cur.merge(ctx,on='candidate_id',how='left',validate='one_to_one')
 c=pd.DataFrame({'candidate_id':cur.candidate_id,'__ts__':cur['__ts__'],'side_name':cur.side_name,'lineage':'current_2026',
  'selection_score':cur.mapped_direct_net,'raw_score':cur['catboost__residual__without_hpo__all_features'],'mapped_score':cur.mapped_direct_net,'alpha_score':cur.base_oof_score,'alpha_target':np.nan,
  'gross':cur.execution_gross_ev_12h,'cost':cur.execution_cost_return,'net':cur.execution_net_ev_12h,'exit_reason':cur.execution_exit_reason,
  'opportunity':cur.opportunity_gross_above_cost_0bps.astype(bool),'map_reference_rows':cur.map_reference_rows,'mapping_kind':'causal_21d_isotonic'}).loc[cur.mapped_eligible].copy()
 out=pd.concat([a,b,c],ignore_index=True)
 out['__ts__']=pd.to_datetime(out['__ts__'],utc=True);out['month']=out['__ts__'].dt.strftime('%Y-%m');out['week']=out['__ts__'].dt.to_period('W-SUN').astype(str);out['era']=out['__ts__'].dt.year.astype(str)
 return out

def _context(data:pd.DataFrame,art:Path)->pd.DataFrame:
 state=pd.read_parquet(art/'regime_episode_ledger_2022_2026_20260730_v1/hourly_state_calendar.parquet',columns=['source_utc','target__pooled_state'])
 phase=pd.read_parquet(art/'transition_pattern_catalogue_20260730_v6/adaptive_phase_labels.parquet',columns=['source_utc','target__pattern_phase'])
 state.source_utc=pd.to_datetime(state.source_utc,utc=True);phase.source_utc=pd.to_datetime(phase.source_utc,utc=True)
 out=data.merge(state.rename(columns={'source_utc':'__ts__','target__pooled_state':'regime_state'}),on='__ts__',how='left')
 out=out.merge(phase.rename(columns={'source_utc':'__ts__','target__pattern_phase':'transition_phase'}),on='__ts__',how='left')
 out['regime_state']=out.regime_state.fillna('unavailable').astype(str);out['transition_phase']=out.transition_phase.fillna('unavailable_or_expost_absent').astype(str)
 return out

def _mapping(data:pd.DataFrame)->pd.DataFrame:
 x=data.dropna(subset=['raw_score','mapped_score']).copy();rows=[]
 for (lineage,month),g in x.groupby(['lineage','month']):
  raw=global_topk(g,score='raw_score',group=(),fraction=.1);mapped=global_topk(g,score='mapped_score',group=(),fraction=.1)
  rows.append({'lineage':lineage,'month':month,'rows':len(g),'raw_mapped_spearman':g.raw_score.corr(g.mapped_score,method='spearman'),'top10_overlap':(raw&mapped).sum()/max(1,mapped.sum()),'raw_selected_net_bps':g.loc[raw,'net'].mean()*1e4,'mapped_selected_net_bps':g.loc[mapped,'net'].mean()*1e4,'mapped_minus_raw_bps':(g.loc[mapped,'net'].mean()-g.loc[raw,'net'].mean())*1e4,'median_reference_rows':g.map_reference_rows.median()})
 return pd.DataFrame(rows)

def run(*,artifacts:Path=ART,output:Path=OUT)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 data=_context(_read_sources(artifacts),artifacts)
 data['selected_global_top10_month']=global_topk(data,score='selection_score')
 data['score_decile']=data.groupby(['lineage','month'])['selection_score'].rank(pct=True,method='first').mul(10).sub(1e-9).floordiv(1).clip(0,9).astype('Int64')
 selected=data.loc[data.selected_global_top10_month].copy()
 outputs={
  'selected_era_month_week.csv':_summary(selected,['lineage','era','month','week'],'global_top10_within_month_week_attribution'),
  'selected_side.csv':_summary(selected,['lineage','era','side_name'],'global_top10_within_month_side_attribution'),
  'selected_regime.csv':_summary(selected,['lineage','era','regime_state'],'global_top10_within_month_regime_attribution'),
  'selected_transition_phase.csv':_summary(selected,['lineage','era','transition_phase'],'global_top10_within_month_expost_phase_attribution'),
  'selected_score_decile.csv':_summary(selected,['lineage','score_decile'],'global_top10_within_month_score_decile_attribution'),
  'selected_opportunity.csv':_summary(selected,['lineage','opportunity'],'global_top10_within_month_opportunity_attribution'),
  'selected_exit_reason.csv':_summary(selected.dropna(subset=['exit_reason']),['lineage','exit_reason'],'global_top10_within_month_exit_attribution'),
  'mapping_rank_diagnostics.csv':_mapping(data),
 }
 # Cause ledger is deliberately evidence-ranked, not a causal claim.
 cause=[]
 for lineage,g in selected.groupby('lineage'):
  alpha=g.dropna(subset=['alpha_score','alpha_target']);ic=alpha.alpha_score.corr(alpha.alpha_target,method='spearman') if len(alpha)>=8 else np.nan
  cause += [
   {'lineage':lineage,'hypothesis':'target_economic_mismatch','evidence_metric':'selected_alpha_rank_ic_vs_net','value':ic,'net_bps':g.net.mean()*1e4,'assessment':'supported' if pd.notna(ic) and ic>0.05 and g.net.mean()<0 else 'not_identifiable_or_not_supported'},
   {'lineage':lineage,'hypothesis':'cost_burden','evidence_metric':'gross_minus_net_bps','value':g.cost.mean()*1e4,'net_bps':g.net.mean()*1e4,'assessment':'supported' if g.gross.mean()>0 and g.net.mean()<0 else 'not_supported'},
   {'lineage':lineage,'hypothesis':'poor_opportunity_capture','evidence_metric':'selected_opportunity_rate','value':g.opportunity.mean(),'net_bps':g.net.mean()*1e4,'assessment':'supported' if g.opportunity.mean()<.5 and g.net.mean()<0 else 'not_supported'},
   {'lineage':lineage,'hypothesis':'sparse_recent_calibration','evidence_metric':'selected_median_map_reference_rows','value':g.map_reference_rows.median(),'net_bps':g.net.mean()*1e4,'assessment':'supported' if g.map_reference_rows.notna().any() and g.map_reference_rows.median()<1000 else 'not_supported_or_not_applicable'},
  ]
 cause=pd.DataFrame(cause)
 rec=pd.DataFrame([
  {'next_ablation':'Cost-aware opportunity hurdle before EV mapping','evidence_basis':'Run only where selected gross is positive while net is negative or opportunity rate is low; measure global-top10 net and day-cluster uncertainty.'},
  {'next_ablation':'Mapping rank-preservation and support shrinkage sweep','evidence_basis':'Use causal maps only; compare raw/mapped top10 overlap and mapped-minus-raw exact net by month without per-side/timestamp quotas.'},
  {'next_ablation':'Side/state/phase diagnostic interaction','evidence_basis':'Only pre-register interactions for groups with adequate selected-day support and negative CI; phase remains ex-post analysis, not a feature.'},
  {'next_ablation':'No regime gate from this audit','evidence_basis':'Regime/phase slices are descriptive and some history is counterfactual or separate lineage.'},
 ])
 output.mkdir(parents=True)
 for n,f in outputs.items():f.to_csv(output/n,index=False)
 cause.to_csv(output/'failure_hypothesis_ledger.csv',index=False);rec.to_csv(output/'evidence_backed_next_ablations.csv',index=False)
 # Latest coverage is explicitly taken from the performance calendar, because detailed mapping ends July 19.
 perf=pd.read_parquet(artifacts/'stack_performance_calendar_2022_2026_20260730_v3/performance_period_metrics.parquet')
 perf.loc[perf.period_type.eq('month') & perf.period.astype(str).str.startswith('2026')].to_csv(output/'latest_period_coverage_and_uncertainty.csv',index=False)
 inputs={'reconstructed':artifacts/'reconstructed_base_residual_stack_2022_2024_20260730_v3/manifest.json','historical_mapping':artifacts/'historical_causal_score_economics_mapping_20260729_v1/canonical_residual__score_residual_expected_ev/manifest.json','current_mapping':artifacts/'current_exact_policy_global_book_mapping_source_20260730_v2/manifest.json','calendar':artifacts/'stack_performance_calendar_2022_2026_20260730_v3/manifest.json','catalogue':artifacts/'transition_pattern_catalogue_20260730_v6/manifest.json'}
 manifest={'schema':'alpha_execution_ev_gap_diagnostic_v1','research_only':True,'promotion_eligible':False,'selection_contract':'one pooled global top10 within reporting month across sides and timestamps; all other dimensions are attribution only','phase_contract':'transition phase is ex-post descriptive only and never an input or selection quota','lineage_limitations':'2022-24 frozen/current spread counterfactual and inverse population; 2025 canonical mapping; 2026 current mapping. Do not pool as deployment PnL.','counts':{'all_candidate_rows':len(data),'selected_rows':len(selected),'lineages':data.lineage.value_counts().to_dict(),'latest_detailed_candidate_month':str(data.month.max())},'inputs_sha256':{k:_sha(v) for k,v in inputs.items()},'outputs_sha256':{p.name:_sha(p) for p in output.iterdir() if p.is_file()}}
 _json(output/'manifest.json',manifest);(output/'manifest.sha256').write_text(f"{_sha(output/'manifest.json')}  manifest.json\n")
 return manifest

def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,default=ART);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args(argv);print(json.dumps(_safe(run(artifacts=a.artifacts,output=a.output)),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
