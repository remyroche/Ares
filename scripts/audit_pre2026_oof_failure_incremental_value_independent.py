#!/usr/bin/env python3
"""Independent, pre-2026-only review of failure/incremental-value OOF study.

This audit never reads a 2026 score, target, economics, or application file.
It checks seals, label/cadence/target invariants, leave-era OOF partitioning,
side-local learner versus globally pooled-top10 semantics, and records the
safeguards required before any separately frozen 2026 scoring run.
"""
from __future__ import annotations
import hashlib,json,math,os,shutil,tempfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts'
SRC=ART/'pre2026_oof_model_failure_incremental_value_20260730_v3';CAD=ART/'pre2026_oof_model_failure_incremental_value_20260730_v4'
OUT=ART/'pre2026_oof_model_failure_incremental_value_independent_review_20260730_v1';TOP=.10
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p:Path,x:object):
 q=p.with_name('.'+p.name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def sealed(p:Path)->bool:return p.exists() and sha(p/'manifest.json')==(p/'manifest.sha256').read_text().split()[0]

def run(output:Path=OUT)->Path:
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 if not sealed(SRC) or not sealed(CAD):raise RuntimeError('unsealed source or cadence supplement')
 x=pd.read_parquet(SRC/'materialized_targets.parquet');x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['execution_label_end_utc']=pd.to_datetime(x.execution_label_end_utc,utc=True)
 required={'candidate_id','__ts__','side_name','source','era','execution_label_end_utc','execution_net_ev_12h','base_score','residual_score','base_selected_global_top10','residual_selected_global_top10','incremental_selected_book_utility','residual_selected_net_failure','residual_selected_false_positive_severity'}
 if missing:=required-set(x):raise RuntimeError(f'missing materialized target fields: {missing}')
 invariants={
  'all_pre2026_label_end':bool(x.execution_label_end_utc.lt(pd.Timestamp('2026-01-01',tz='UTC')).all()),
  'all_label_end_after_decision':bool(x.execution_label_end_utc.gt(x.__ts__).all()),
  'all_hourly_timestamp':bool((x.__ts__.astype('int64')%pd.Timedelta(hours=1).value==0).all()),
  'unique_candidate_identity':bool(not x.candidate_id.duplicated().any()),
  'finite_economics_and_scores':bool(x[['execution_net_ev_12h','base_score','residual_score']].notna().all().all()),
  'no_2026_candidate_rows':bool(x.__ts__.dt.year.lt(2026).all()),
 }
 selection=[]
 for era,g in x.groupby('era',sort=True):
  expected=math.ceil(len(g)*TOP);base=g.sort_values(['base_score','candidate_id'],ascending=[False,True],kind='stable').index[:expected];resid=g.sort_values(['residual_score','candidate_id'],ascending=[False,True],kind='stable').index[:expected]
  selection.append({'era':era,'rows':len(g),'expected_global_top10_rows':expected,'base_selected_rows':int(g.base_selected_global_top10.sum()),'residual_selected_rows':int(g.residual_selected_global_top10.sum()),'base_exact_global_score_membership':bool(g.loc[base,'base_selected_global_top10'].all() and int(g.base_selected_global_top10.sum())==expected),'residual_exact_global_score_membership':bool(g.loc[resid,'residual_selected_global_top10'].all() and int(g.residual_selected_global_top10.sum())==expected),'both_sides_in_base_book':int(g.loc[g.base_selected_global_top10,'side_name'].nunique())==2,'both_sides_in_residual_book':int(g.loc[g.residual_selected_global_top10,'side_name'].nunique())==2})
 selection=pd.DataFrame(selection)
 y=x.execution_net_ev_12h;delta=x.residual_selected_global_top10.astype(int)-x.base_selected_global_top10.astype(int)
 target_checks={'incremental_utility_exact':bool((x.incremental_selected_book_utility-(y*delta)).abs().fillna(0).max()<1e-12),'failure_only_residual_global_top10':bool((x.residual_selected_net_failure.eq(1)<=x.residual_selected_global_top10).all()),'failure_net_sign_exact':bool(x.residual_selected_net_failure.eq((x.residual_selected_global_top10&y.le(0)).astype(int)).all()),'severity_only_residual_global_top10':bool(x.loc[~x.residual_selected_global_top10,'residual_selected_false_positive_severity'].isna().all()),'severity_exact':bool((x.loc[x.residual_selected_global_top10,'residual_selected_false_positive_severity']-(-y.loc[x.residual_selected_global_top10]).clip(lower=0)).abs().fillna(0).max()<1e-12)}
 # Read OOF outputs only; each has a pre-2026 held-era label and must be
 # one prediction per candidate for that target/arm (no repeated fold rows).
 oof=[]
 for p in sorted(SRC.glob('leave_era_oof_*.parquet')):
  q=pd.read_parquet(p,columns=['candidate_id','__ts__','side_name','era','target','arm','prediction','actual_target','residual_selected_global_top10']);q['__ts__']=pd.to_datetime(q.__ts__,utc=True)
  oof.append({'file':p.name,'rows':len(q),'unique_candidate_rows':int(q.candidate_id.nunique()),'duplicates':int(q.candidate_id.duplicated().sum()),'all_pre2026':bool(q.__ts__.dt.year.lt(2026).all()),'all_hourly':bool((q.__ts__.astype('int64')%pd.Timedelta(hours=1).value==0).all()),'both_sides':bool(set(q.side_name)=={'long','short'}),'failure_target_selected_only':bool(q.loc[q.target.eq('selected_net_failure'),'residual_selected_global_top10'].all()),'prediction_finite':bool(q.prediction.notna().all())})
 oof=pd.DataFrame(oof)
 metrics=pd.read_csv(SRC/'fold_metrics.csv');side=[]
 for target in ['incremental_selected_book_utility','selected_net_failure','top_tail_false_positive_severity']:
  threshold=.5 if target=='selected_net_failure' else 0.
  for (arm,scope),g in metrics[(metrics.target==target)&metrics.scope.isin(['pooled','long','short'])].groupby(['arm','scope'],sort=True):
   v=g.rank_metric.dropna();side.append({'target':target,'arm':arm,'scope':scope,'held_eras':len(v),'median_rank_metric':float(v.median()),'min_rank_metric':float(v.min()),'positive_fold_fraction':float((v>threshold).mean())})
 side=pd.DataFrame(side)
 stability=pd.read_csv(SRC/'stability_summary.csv');cadence=pd.read_csv(CAD/'cadence_provenance_audit.csv')
 # Structural OOF is valid label holdout, but all leave-era fits deliberately
 # include later calendar eras. It is therefore not a chronological estimate.
 review={'sealed_v3':True,'sealed_cadence_v4':True,'no_2026_economics_inspected':True,'direct_leakage_or_target_mismatch_detected':not(all(invariants.values()) and all(target_checks.values()) and (oof.duplicates.eq(0)&oof.all_pre2026&oof.all_hourly&oof.prediction_finite).all()),'leave_era_oof_integrity':'PASS_FOR_ERA_HOLDOUT_BUT_NOT_CHRONOLOGICAL: code and fold contract exclude the held era, while fitting on every other era, including later eras; this is structural OOF not a deployment-time forward validation.','global_selection_semantics':'PASS_AS_IMPLEMENTED: one score-ranked 10% book is pooled across both sides for each whole fixed era. This is not a causal per-decision or rolling-live threshold; it must not be silently treated as one global historical book or an intraday policy label.','passing_metrics_interpretation':'Utility and selected-failure arms pass the predeclared pooled era-holdout gate. False-positive severity fails every arm. Passing utility is partly mechanical because base/residual scores define book membership and are features; it supports a selected-book overlay hypothesis, not admission/reranking of the full universe.','frozen_2026_application_verdict':'CONDITIONAL_SCORE_ONLY_ALLOWED_NOT_PROMOTION: a separately sealed all-pre2026 refit may score untouched 2026 candidates, but no outcome-based selection, calibration, threshold tuning, economics claim, policy change, or portfolio replay is justified by this study alone.'}
 safeguards=[
  {'priority':1,'safeguard':'Freeze one arm/direction/threshold before opening 2026 outcomes; do not select a winner from the four observed arms using 2026.'},
  {'priority':2,'safeguard':'Use selected_net_failure only as a risk overlay after the existing base/residual score has produced the predeclared pooled global top10 book; do not use utility to admit new candidates or rerank the full universe.'},
  {'priority':3,'safeguard':'Define and seal a causal 2026 pooled-global-top10 reference/tie-break and rebalance horizon. The historical target ranks over entire eras and is not itself live-causal.'},
  {'priority':4,'safeguard':'Refit exactly side-local with frozen feature lists/learner/caps, train only on labels ending before the 2026 scoring cutoff, and emit model/training-row/hash/provenance manifests.'},
  {'priority':5,'safeguard':'Fail closed for missing arm context; preserve both-side pooled book semantics and report overlay impact separately by side.'},
  {'priority':6,'safeguard':'Treat leave-era results as nonchronological. Require a later chronological/OOS evaluation before promotion; retain all 2026 economics untouched until the preregistered score-only application is sealed.'},
  {'priority':7,'safeguard':'Do not deploy false-positive-severity as a decision score: it fails its predeclared stability gate.'},
 ]
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  pd.DataFrame([invariants|target_checks]).to_csv(stage/'invariant_target_audit.csv',index=False);selection.to_csv(stage/'global_top10_semantics_audit.csv',index=False);oof.to_csv(stage/'leave_era_oof_output_audit.csv',index=False);side.to_csv(stage/'side_and_pooled_metric_review.csv',index=False);stability.to_csv(stage/'source_stability_summary.csv',index=False);cadence.to_csv(stage/'cadence_supplement_reference.csv',index=False);pd.DataFrame(safeguards).to_csv(stage/'required_frozen_2026_safeguards.csv',index=False);dump(stage/'review.json',review)
  contract={'scope':'independent review of v3/v4 pre-2026 artifacts only; no 2026 score/economics/application input opened','cadence':'1h candidate/model rows; 1m remains nested label evidence only','review_limit':'does not create a 2026 model or application; verifies prerequisites and records safeguards only'};dump(stage/'contract.json',contract)
  files=[p for p in stage.iterdir() if p.is_file()];m={'schema':'pre2026_oof_model_failure_incremental_value_independent_review_v1','status':'SEALED_CONDITIONAL_FROZEN_2026_SCORE_ONLY_REVIEW','promotion_eligible':False,'review':review,'contract':contract,'inputs_sha256':{str((SRC/'manifest.json').resolve()):sha(SRC/'manifest.json'),str((SRC/'materialized_targets.parquet').resolve()):sha(SRC/'materialized_targets.parquet'),str((CAD/'manifest.json').resolve()):sha(CAD/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
