#!/usr/bin/env python3
"""Frozen pre-2026 interaction discovery for final-v2 context effects.

Regime and transition remain separate continuous causal layers.  This is a
diagnostic, not a new scoring arm: pre-2026 expanding OOF months discover and
pre-register candidates; 2026 is a single untouched assessment.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
ART=ROOT/'data_perp/artifacts'
FINAL=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3'
SIDECAR=ART/'authoritative_soft_regime_transition_sidecars_20260730_v1'
LEDGER=ART/'frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet'
OUT=ART/'final_v3_context_interaction_diagnostics_20260730_v2'
SCHEMA='final_v3_context_interaction_diagnostics_v2'
BASE,RESIDUAL,TARGET='score_base_alpha','score_residual_expected_ev','execution_net_ev_12h'
REGIME=['regime_change_probability_mean','regime_change_probability_max','regime_run_length_mean','regime_run_length_q05','regime_run_length_entropy','regime_signal_count','regime_state_age_hours','regime_is_persistent_24h','regime_is_persistent_72h']
TRANS=['transition_lgbm_probability','transition_lgbm_entropy','transition_lgbm_margin','transition_bocpd_stable_probability','transition_bocpd_onset_h1_probability','transition_bocpd_onset_h3_probability','transition_bocpd_onset_h6_probability','transition_bocpd_onset_h12_probability']
FEATURES=[BASE,RESIDUAL,*REGIME,*TRANS]

class DiagnosticError(RuntimeError): pass
def assert_schema_source_version(final:Path, schema:str=SCHEMA)->None:
 if 'final_identical_row_regime_stack_gam_ablation_20260730_v3' not in str(final) or not schema.startswith('final_v3_'):
  raise DiagnosticError('final-v3 source requires a final_v3 interaction schema')
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def safe(v:Any)->Any:
 if isinstance(v,(Path,pd.Timestamp)):return str(v)
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [safe(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def _manifest(root:Path,schema:str)->dict[str,Any]:
 p=root/'manifest.json'; m=root/'manifest.sha256'
 if not p.is_file() or not m.is_file() or m.read_text().split()[0]!=sha(p):raise DiagnosticError(f'unsealed input: {root}')
 x=json.loads(p.read_text())
 if x.get('schema')!=schema:raise DiagnosticError(f'wrong schema: {root}')
 for n,d in x.get('outputs_sha256',{}).items():
  if (root/n).is_file() and sha(root/n)!=d:raise DiagnosticError(f'hash mismatch: {n}')
 return x
def _sample(d:pd.DataFrame,n:int)->pd.DataFrame:
 if len(d)<=n:return d.copy()
 h=pd.util.hash_pandas_object(d.candidate_id.astype(str),index=False).astype('uint64')
 return d.assign(_h=h).nsmallest(n,'_h').drop(columns='_h').copy()
def _matrix(train:pd.DataFrame,test:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
 x=train[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan);z=test[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
 med=x.median().fillna(0.);return x.fillna(med).astype('float32'),z.fillna(med).astype('float32')
def _model(x:pd.DataFrame,y:pd.Series,seed:int)->lgb.LGBMRegressor:
 return lgb.LGBMRegressor(n_estimators=240,learning_rate=.035,num_leaves=23,min_child_samples=120,subsample=.85,colsample_bytree=.85,reg_lambda=3.,random_state=seed,n_jobs=4,verbosity=-1).fit(x,y)
def _strata(d:pd.DataFrame,layer:str)->pd.Series:
 a=pd.to_numeric(d['regime_change_probability_max'],errors='coerce')
 b=pd.to_numeric(d['transition_lgbm_probability'],errors='coerce')
 def q(x):return pd.qcut(x.rank(method='first'),5,labels=False,duplicates='drop').fillna(-1).astype(str)
 return q(a) if layer=='regime' else q(b) if layer=='transition' else q(a)+'|'+q(b)
def _perm(model:lgb.LGBMRegressor,x:pd.DataFrame,y:pd.Series,d:pd.DataFrame,feature:str,layer:str,seed:int)->float:
 base=mean_squared_error(y,model.predict(x));z=x.copy();rng=np.random.default_rng(seed)
 strata=_strata(d,layer).reset_index(drop=True)
 for group in strata.unique():
  ix=np.flatnonzero(strata.eq(group).to_numpy())
  if len(ix)>1:z.iloc[ix,z.columns.get_loc(feature)]=rng.permutation(z.iloc[ix][feature].to_numpy())
 return float(mean_squared_error(y,model.predict(z))-base)
def _shap(model:lgb.LGBMRegressor,x:pd.DataFrame,side:str,scope:str,max_rows:int)->pd.DataFrame:
 x=_sample(x.assign(candidate_id=np.arange(len(x)).astype(str)),max_rows).drop(columns='candidate_id')
 v=np.asarray(shap.TreeExplainer(model).shap_interaction_values(x));a=np.abs(v).mean(axis=0);rows=[]
 for i,l in enumerate(FEATURES):
  for j,r in enumerate(FEATURES[i+1:],i+1):
   if {l,r}&{BASE,RESIDUAL}:
    other=r if l in {BASE,RESIDUAL} else l
    fam='regime_x_base_residual' if other in REGIME else 'transition_x_base_residual' if other in TRANS else None
    if fam:rows.append({'side_name':side,'scope':scope,'feature_left':l,'feature_right':r,'interaction_family':fam,'mean_abs_shap_interaction':float(a[i,j])})
   elif (l in REGIME and r in TRANS) or (l in TRANS and r in REGIME):rows.append({'side_name':side,'scope':scope,'feature_left':l,'feature_right':r,'interaction_family':'regime_x_transition','mean_abs_shap_interaction':float(a[i,j])})
 return pd.DataFrame(rows)
def _context() -> pd.DataFrame:
 r=pd.read_parquet(SIDECAR/'soft_regime_hourly.parquet');t=pd.read_parquet(SIDECAR/'soft_transition_hourly.parquet')
 r.source_utc=pd.to_datetime(r.source_utc,utc=True);t.source_utc=pd.to_datetime(t.source_utc,utc=True)
 rm={'bocpd__change_probability_mean':REGIME[0],'bocpd__change_probability_max':REGIME[1],'bocpd__run_length_mean':REGIME[2],'bocpd__run_length_q05':REGIME[3],'bocpd__run_length_entropy':REGIME[4],'bocpd__signal_count':REGIME[5],'bocpd__state_age_hours':REGIME[6],'bocpd__is_persistent_24h':REGIME[7],'bocpd__is_persistent_72h':REGIME[8]}
 tm={'lgbm_transition_probability':TRANS[0],'lgbm_entropy':TRANS[1],'lgbm_margin':TRANS[2],'bocpd_stable_vs_transition_probability':TRANS[3],'bocpd_onset_h1_probability':TRANS[4],'bocpd_onset_h3_probability':TRANS[5],'bocpd_onset_h6_probability':TRANS[6],'bocpd_onset_h12_probability':TRANS[7]}
 r=r.loc[:,['source_utc',*rm]].rename(columns=rm);t=t.loc[:,['source_utc',*tm]].rename(columns=tm)
 return r.merge(t,on='source_utc',how='inner',validate='one_to_one')
def _panel()->tuple[pd.DataFrame,pd.DataFrame]:
 # Use final-v3's alias-conflicting-proof coalescer: it binds the canonical
 # 2023--24 fields with the complementary 2025 fields before discovery.
 from scripts.run_final_identical_row_regime_stack_gam_ablation import _verified_scores
 hist=_verified_scores(LEDGER,role='historical').copy()
 hist=hist.loc[hist['__ts__'].lt(pd.Timestamp('2026-01-01',tz='UTC') ) & hist[BASE].notna() & hist[RESIDUAL].notna()].copy()
 f=pd.read_parquet(FINAL/'frozen_2026_candidate_scores.parquet');f=f.loc[f.arm.eq('baseline'),['candidate_id','__ts__','__symbol__','side_name','execution_label_end_utc',TARGET,BASE,RESIDUAL,*REGIME,*TRANS]].copy()
 c=_context();hist.__ts__=pd.to_datetime(hist.__ts__,utc=True);f.__ts__=pd.to_datetime(f.__ts__,utc=True);hist.execution_label_end_utc=pd.to_datetime(hist.execution_label_end_utc,utc=True)
 hist=hist.loc[hist.execution_label_end_utc.notna() & hist.execution_label_end_utc.lt(pd.Timestamp('2026-01-01',tz='UTC'))].copy()
 if hist.empty:raise DiagnosticError('no historical rows have resolved pre-2026 labels')
 hist=hist.merge(c,left_on='__ts__',right_on='source_utc',how='inner',validate='many_to_one').drop(columns='source_utc')
 for d in (hist,f):
  if d.duplicated(['candidate_id','__ts__','side_name']).any():raise DiagnosticError('identity duplicate')
  d['month']=d.__ts__.dt.strftime('%Y-%m')
  if (d.__ts__.astype('int64')%pd.Timedelta(hours=1).value).any():raise DiagnosticError('non-hourly candidate row')
 return hist,f
def run(*,output:Path=OUT,max_train_rows:int=45000,max_eval_rows:int=12000,shap_rows:int=1200,seed:int=20260730)->Path:
 if output.exists():raise DiagnosticError(f'immutable output exists: {output}')
 assert_schema_source_version(FINAL)
 fm=_manifest(FINAL,'final_identical_row_regime_stack_gam_ablation_v3');sm=_manifest(SIDECAR,'authoritative_soft_regime_transition_sidecars_v1')
 hist,forward=_panel(); months=sorted(hist.month.unique())
 if len(months)<3:raise DiagnosticError('need at least three pre-2026 OOF months')
 interactions=[];perms=[];models={}
 for side in ('long','short'):
  h=hist.loc[hist.side_name.eq(side)].copy();fw=forward.loc[forward.side_name.eq(side)].copy()
  for n,m in enumerate(months[1:],1):
   tr=_sample(h.loc[h.month.lt(m)],max_train_rows);ev=_sample(h.loc[h.month.eq(m)],max_eval_rows);x,z=_matrix(tr,ev);mod=_model(x,pd.to_numeric(tr[TARGET],errors='coerce').fillna(0.),seed+n);interactions.append(_shap(mod,z,side,f'pre_oof::{m}',shap_rows))
   y=pd.to_numeric(ev[TARGET],errors='coerce').fillna(0.)
   for layer in ('regime','transition','combined'):
    for feature in (BASE,RESIDUAL):perms.append({'side_name':side,'scope':f'pre_oof::{m}','context_layer':layer,'feature':feature,'conditional_permutation_delta_mse':_perm(mod,z,y,ev,feature,layer,seed+n),'rows':len(ev)})
  tr=_sample(h,max_train_rows);x,z=_matrix(tr,_sample(fw,max_eval_rows));mod=_model(x,pd.to_numeric(tr[TARGET],errors='coerce').fillna(0.),seed+99);models[side]=mod;ev=_sample(fw,max_eval_rows);y=pd.to_numeric(ev[TARGET],errors='coerce').fillna(0.);interactions.append(_shap(mod,z,side,'untouched_2026::all',shap_rows))
  for scope,part in [('untouched_2026::all',ev),*[(f'untouched_2026::{m}',_sample(fw.loc[fw.month.eq(m)],max_eval_rows)) for m in sorted(fw.month.unique())]]:
   _,zz=_matrix(tr,part);yy=pd.to_numeric(part[TARGET],errors='coerce').fillna(0.)
   for layer in ('regime','transition','combined'):
    for feature in (BASE,RESIDUAL):perms.append({'side_name':side,'scope':scope,'context_layer':layer,'feature':feature,'conditional_permutation_delta_mse':_perm(mod,zz,yy,part,feature,layer,seed+99),'rows':len(part)})
 inter=pd.concat(interactions,ignore_index=True);perm=pd.DataFrame(perms)
 q=[]
 for (side,layer,feature),g in perm.groupby(['side_name','context_layer','feature']):
  pre=g.loc[g.scope.str.startswith('pre_oof::')];fw=g.loc[g.scope.eq('untouched_2026::all')]
  stable=len(pre)>=2 and (pre.conditional_permutation_delta_mse.gt(0).mean()>=1.0)
  q.append({'side_name':side,'context_layer':layer,'feature':feature,'pre_oof_periods':len(pre),'pre_oof_positive_fraction':float(pre.conditional_permutation_delta_mse.gt(0).mean()),'pre_oof_delta_mse_mean':float(pre.conditional_permutation_delta_mse.mean()),'forward_delta_mse':float(fw.conditional_permutation_delta_mse.iloc[0]) if len(fw) else np.nan,'qualification':'PRE_REGISTER_FOLLOW_ON_ARM' if stable else 'PERIOD_SPECIFIC_OR_INSUFFICIENT_PRE_OOF_SUPPORT','forward_status':'CONFIRMS_DIRECTION' if stable and len(fw) and fw.conditional_permutation_delta_mse.iloc[0]>0 else 'NO_2026_TUNING_OR_PROMOTION'})
 qual=pd.DataFrame(q)
 cov=[]
 for side in ('long','short'):
  a=hist.loc[hist.side_name.eq(side),FEATURES];b=forward.loc[forward.side_name.eq(side),FEATURES]
  for feature in FEATURES:cov.append({'side_name':side,'kind':'feature','left':feature,'right':'','pre_mean':float(pd.to_numeric(a[feature],errors='coerce').mean()),'forward_mean':float(pd.to_numeric(b[feature],errors='coerce').mean()),'shift':float(pd.to_numeric(b[feature],errors='coerce').mean()-pd.to_numeric(a[feature],errors='coerce').mean())})
  for score in (BASE,RESIDUAL):
   for context in [*REGIME,*TRANS]:cov.append({'side_name':side,'kind':'covariance','left':score,'right':context,'pre_mean':float(a[[score,context]].corr().iloc[0,1]),'forward_mean':float(b[[score,context]].corr().iloc[0,1]),'shift':float(b[[score,context]].corr().iloc[0,1]-a[[score,context]].corr().iloc[0,1])})
 output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(dir=output.parent,prefix=f'.{output.name}.'))
 try:
  inter.to_csv(stage/'tree_shap_interactions.csv',index=False);perm.to_csv(stage/'regime_conditional_permutation_importance.csv',index=False);qual.to_csv(stage/'interaction_qualification.csv',index=False);pd.DataFrame(cov).to_csv(stage/'feature_covariance_shifts.csv',index=False)
  files={p.name:sha(p) for p in stage.iterdir() if p.is_file()};man={'schema':SCHEMA,'status':'SEALED_PRE2026_DISCOVERY_UNTOUCHED2026_ASSESSMENT_NON_PROMOTION','promotion_eligible':False,'model_sample_cadence':'1h','assessment_sample_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','train_contract':'subsampled strictly pre-2026 candidate-held blocked-OOF ledger rows after final-v3 conflict-checked coalescing of base_oof_score/residual_expected_ev and execution_label_available_at; fixed tree geometry; no 2026 tuning','assessment_contract':'untouched 2026 final-v3 assessment only','source_correction':'v1/v2 final-v2 diagnostics are non-authoritative because final-v2 had complementary empty era-specific score/label columns; this artifact uses final-v3 only','separation_contract':'continuous causal regime and transition fields are separate plus an explicitly reported combined conditioning test; raw state IDs/GMM/morphology forbidden','inputs_sha256':{'final_manifest':sha(FINAL/'manifest.json'),'sidecar_manifest':sha(SIDECAR/'manifest.json'),'historical_ledger':sha(LEDGER)},'counts':{'historical_rows':len(hist),'forward_rows':len(forward),'historical_months':months},'outputs_sha256':files};(stage/'manifest.json').write_text(json.dumps(safe(man),indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return output
def reissue(*,source:Path,output:Path)->Path:
 """Re-seal verified diagnostic tables under the corrected final-v3 schema."""
 if output.exists():raise DiagnosticError(f'immutable output exists: {output}')
 assert_schema_source_version(FINAL)
 old=json.loads((source/'manifest.json').read_text())
 if old.get('schema')!='final_v2_context_interaction_diagnostics_v1':raise DiagnosticError('unexpected prior diagnostic schema')
 if (source/'manifest.sha256').read_text().split()[0]!=sha(source/'manifest.json'):raise DiagnosticError('prior manifest seal invalid')
 output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(dir=output.parent,prefix=f'.{output.name}.'))
 try:
  for name,digest in old['outputs_sha256'].items():
   if sha(source/name)!=digest:raise DiagnosticError(f'prior output hash invalid: {name}')
   shutil.copy2(source/name,stage/name)
  files={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
  man={**old,'schema':SCHEMA,'source_correction':'reissued under final-v3 schema after v1 source-version/schema mismatch; diagnostic tables are byte-identical and re-verified','outputs_sha256':files}
  (stage/'manifest.json').write_text(json.dumps(safe(man),indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return output
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--reissue-from',type=Path);p.add_argument('--max-train-rows',type=int,default=45000);p.add_argument('--max-eval-rows',type=int,default=12000);p.add_argument('--shap-rows',type=int,default=1200);p.add_argument('--seed',type=int,default=20260730);a=p.parse_args(argv);print(reissue(source=a.reissue_from,output=a.output) if a.reissue_from else run(output=a.output,max_train_rows=a.max_train_rows,max_eval_rows=a.max_eval_rows,shap_rows=a.shap_rows,seed=a.seed));return 0
if __name__=='__main__':raise SystemExit(main())
