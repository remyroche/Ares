#!/usr/bin/env python3
"""Stage-D D3 compact action model, leave-group-out, and D4 causal replay.

The runner is fail-closed on canonical D2 v4.  It fits side-local models using
development-approved groups only, freezes preprocessing/features/calibration
before final OOS, and replays only fixed 0/25/50-bps action margins (never top-k).  It never
changes candidate entry, sizing, concurrency, exposure, or portfolio policy.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.pipeline_supersession import assert_artifact_usable
ART=ROOT/'data_perp/artifacts'
D2=ART/'stage_d_action_mechanism_ablation_20260731_v4'
D2_MANIFEST=D2/'run_manifest.json';D2_SELECTION=D2/'stage_d_d9_development_selection.json'
FEATURE_ROOT=ART/'stage_d_action_features_20260731_v5';FEATURES=FEATURE_ROOT/'stage_d_action_features.parquet';GROUPS=FEATURE_ROOT/'stage_d_action_feature_groups.json'
FEATURE_RUN_MANIFEST=FEATURE_ROOT/'run_manifest.json';FEATURE_LINEAGE=FEATURE_ROOT/'stage_d_action_feature_lineage.parquet'
TARGETS=ART/'stage_d_action_counterfactuals_20260731_v2/stage_d_action_counterfactuals.parquet'
DEFAULT_OUTPUT=ART/'stage_d_compact_action_model_20260731_v9'
SCHEMA='stage_d_compact_action_model_v9';SEED=20260731;HORIZON=pd.Timedelta(hours=12);SIDES=('long','short');MARGINS=(0.,25.,50.)
DEV_START=pd.Timestamp('2024-04-01T00:00:00Z');FINAL_START=pd.Timestamp('2024-08-01T00:00:00Z');FINAL_END=pd.Timestamp('2024-12-01T00:00:00Z')
MAX_FEATURES_PER_SIDE=32;MIN_TRAIN=500;BOOTSTRAPS=1000
GROUP_MAP={'A1':'A1_path_geometry_to_clear','A2':'A2_candle_rejection_structure','A4':'A4_volatility_instability_to_clear','A5':'A5_market_cross_sectional_confirmation','A9':'A9_compact_composites'}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p:Path,x:Any)->None:p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def idhash(x:Iterable[Any])->str:return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def train_mask(f:pd.DataFrame,start:pd.Timestamp)->pd.Series:return f.action_decision_ts.lt(start-HORIZON)&f.label_available_ts.lt(start)

def require_d2()->tuple[dict[str,Any],dict[str,Any]]:
 if not D2_MANIFEST.exists() or not D2_SELECTION.exists():raise FileNotFoundError(f'wait for canonical D2 v4: {D2}')
 assert_artifact_usable(D2, purpose='training')
 m=json.loads(D2_MANIFEST.read_text());s=json.loads(D2_SELECTION.read_text())
 # The immutable v4 artifact retained the evaluator's internal v2 schema;
 # canonicality is established by the v4 root plus its byte-identical v5
 # rerun/seals, not by renaming the internal schema string.
 if m.get('schema')!='stage_d_action_mechanism_ablation_v2':raise ValueError('D3 requires the sealed canonical D2 v4 evaluator schema')
 expected=m.get('outputs_sha256',{}).get(D2_SELECTION.name)
 if not expected or sha(D2_SELECTION)!=expected:raise ValueError('D2 v4 development selection seal mismatch')
 if s.get('source')!='development_oof_only':raise ValueError('D2 group selection is not development-only')
 approved=list(s.get('approved',[]))
 if any(g not in GROUP_MAP for g in approved):raise ValueError(f'unsupported D2-approved groups: {approved}')
 return m,s

def load_frame()->tuple[pd.DataFrame,dict[str,list[str]]]:
 x=pd.read_parquet(FEATURES);y=pd.read_parquet(TARGETS)
 outcomes=['candidate_id','net_exit_now_gross_bps','net_exit_now_cost_bps','net_exit_now_bps','net_continue_gross_bps','net_continue_cost_bps','net_continue_bps','delta_continue_bps','continue_better']
 f=x.merge(y[outcomes],on='candidate_id',validate='one_to_one')
 for c in ('entry_ts','action_decision_ts','label_available_ts'):f[c]=pd.to_datetime(f[c],utc=True)
 f=f.sort_values(['action_decision_ts','candidate_id'],kind='stable').reset_index(drop=True)
 if len(f)!=len(x) or f.candidate_id.duplicated().any():raise ValueError('fixed D0 population mismatch')
 if not np.allclose(f.net_continue_bps-f.net_exit_now_bps,f.delta_continue_bps):raise ValueError('paired target drift')
 f['time_to_clear_bucket']=pd.cut(f.time_to_clear_minutes,[-np.inf,15,60,180,360,np.inf],labels=['01-15m','16-60m','61-180m','181-360m','361-718m']).astype(str)
 f['volatility_bucket']=pd.cut(f.realised_volatility,[-np.inf,25,50,100,np.inf],labels=['<=25bps','25-50bps','50-100bps','>100bps']).astype(str)
 groups={k:list(v) for k,v in json.loads(GROUPS.read_text()).items() if k.startswith('A') and isinstance(v,list)}
 return f,groups

def fit_preprocess(train:pd.DataFrame,requested:list[str],target:np.ndarray,seed:int,cap:int=MAX_FEATURES_PER_SIDE)->dict[str,Any]:
 state={'requested':list(dict.fromkeys(requested)),'removed':{},'clip':{},'median':{},'selected':[],'correlation_threshold':.96,'feature_cap':cap}
 usable=[]
 for c in state['requested']:
  if c not in train:state['removed'][c]='absent';continue
  s=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan);nn=s.dropna()
  if s.notna().mean()<.5:state['removed'][c]='availability_below_0.5';continue
  if nn.nunique()<2 or nn.value_counts(normalize=True).iloc[0]>=.995:state['removed'][c]='near_constant';continue
  lo,hi=nn.quantile([.01,.99]);state['clip'][c]=[float(lo),float(hi)];state['median'][c]=float(nn.median());usable.append(c)
 def raw(c:str)->pd.Series:return pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan).clip(*state['clip'][c]).fillna(state['median'][c])
 a=pd.DataFrame({c:raw(c) for c in usable},index=train.index);reps=[];corr=a.corr(method='spearman').abs()
 for c in usable:
  if all(float(corr.loc[c,q])<.96 for q in reps):reps.append(c)
 selected=reps
 if len(reps)>cap:
  sel=lgb.LGBMRegressor(objective='huber',n_estimators=80,num_leaves=31,learning_rate=.05,min_child_samples=80,reg_lambda=2.,random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)
  sel.fit(a[reps],target);gain=pd.Series(sel.booster_.feature_importance('gain'),index=reps);selected=gain.sort_values(ascending=False,kind='stable').head(cap).index.tolist();state['gain']={c:float(gain[c]) for c in selected}
 state['selected']=selected
 return state

def apply_preprocess(frame:pd.DataFrame,state:dict[str,Any])->pd.DataFrame:
 return pd.DataFrame({c:pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan).clip(*state['clip'][c]).fillna(state['median'][c]) for c in state['selected']},index=frame.index)

def fit_model(x:pd.DataFrame,y:pd.Series,seed:int)->lgb.LGBMRegressor:
 m=lgb.LGBMRegressor(objective='huber',alpha=.9,n_estimators=160,num_leaves=31,learning_rate=.04,min_child_samples=80,subsample=.9,colsample_bytree=.9,reg_lambda=2.,random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1);m.fit(x,y);return m

def fit_calibration(dev:pd.DataFrame)->dict[str,Any]:
 if len(dev)<250 or dev.raw_predicted_delta_bps.std()<=1e-9:raise ValueError('insufficient development OOF for frozen calibration')
 slope,intercept=np.polyfit(dev.raw_predicted_delta_bps,dev.delta_continue_bps,1)
 clf=LogisticRegression(C=10.,max_iter=500).fit(dev[['raw_predicted_delta_bps']],dev.continue_better)
 # BLAS reductions can differ at ~1e-13 across otherwise identical runs.
 # Canonicalise well below any economically meaningful precision so the
 # replay and its sealed artifacts are byte-deterministic.
 canonical=lambda value:float(np.round(float(value),10))
 return {'source':'development_oof_only','rows':len(dev),'slope':canonical(slope),'intercept':canonical(intercept),'probability_coef':canonical(clf.coef_[0,0]),'probability_intercept':canonical(clf.intercept_[0])}
def calibrate(raw:np.ndarray,c:dict[str,Any])->tuple[np.ndarray,np.ndarray]:
 raw=np.asarray(raw);mapped=raw*c['slope']+c['intercept'];logit=raw*c['probability_coef']+c['probability_intercept'];return mapped,1/(1+np.exp(-np.clip(logit,-40,40)))

def calibrate_development(dev:pd.DataFrame)->pd.DataFrame:
 parts=[]
 for side in SIDES:
  q=dev[dev.side.eq(side)].copy();cal=fit_calibration(q);q['predicted_delta_continue_bps'],q['predicted_continue_probability']=calibrate(q.raw_predicted_delta_bps.to_numpy(),cal);parts.append(q)
 return pd.concat(parts,ignore_index=True)

def compact_readmission_decision(full_diag:dict[str,Any],candidate_diag:dict[str,Any])->bool:
 calibration_error=lambda q:abs(q['calibration_slope']-1.)+abs(q['calibration_intercept_bps'])/100.
 return bool(candidate_diag['net_policy_bps']>full_diag['net_policy_bps'] and candidate_diag['mae_bps']<full_diag['mae_bps'] and candidate_diag['spearman_ic']>full_diag['spearman_ic'] and calibration_error(candidate_diag)<=calibration_error(full_diag)+1e-9)

def lineage_evidence(frame:pd.DataFrame,included_groups:list[str],features:list[str])->dict[str,Any]:
 manifest=json.loads(FEATURE_RUN_MANIFEST.read_text());lineage=pd.read_parquet(FEATURE_LINEAGE);group_contract=json.loads(GROUPS.read_text())
 output_seals=manifest.get('outputs_sha256',{});seal_checks={p.name:bool(output_seals.get(p.name) and sha(p)==output_seals[p.name]) for p in (FEATURES,GROUPS,FEATURE_LINEAGE)}
 admitted=lineage[lineage.disposition.eq('ADMITTED_CAUSAL')&lineage.point_in_time_safe.eq(True)&lineage.live_reproducible.eq(True)]
 admitted_names=set(admitted.feature_name.astype(str));unadmitted=sorted(set(features)-admitted_names)
 dispositions=group_contract.get('dispositions',{});blocked_ok=all(dispositions.get(g)==expected for g,expected in {'A6':'REJECTED_LINEAGE','A7':'REJECTED_LINEAGE','A8':'REJECTED_OOF_LINEAGE'}.items())
 checks={'feature_pack_output_seals_match':all(seal_checks.values()),'all_included_features_admitted_causal':not unadmitted,'a6_a7_a8_excluded_and_rejected':blocked_ok and not any(g.startswith(('A6_','A7_','A8_')) for g in included_groups),'feature_timestamps_available_by_action_decision':bool(frame.feature_available_ts.le(frame.action_decision_ts).all()),'candidate_ids_unique':not frame.candidate_id.duplicated().any(),'counterfactual_delta_exact':bool(np.allclose(frame.net_continue_bps-frame.net_exit_now_bps,frame.delta_continue_bps)),'cost_once_continue':bool(np.allclose(frame.net_continue_gross_bps-frame.net_continue_cost_bps,frame.net_continue_bps)),'cost_once_exit':bool(np.allclose(frame.net_exit_now_gross_bps-frame.net_exit_now_cost_bps,frame.net_exit_now_bps))}
 return {'passed':all(checks.values()),'checks':checks,'feature_pack_seals':seal_checks,'included_groups':included_groups,'included_feature_count':len(features),'unadmitted_features':unadmitted,'blocked_group_dispositions':{g:dispositions.get(g) for g in ('A6','A7','A8')}}

def fold_score(frame:pd.DataFrame,features:list[str],arm:str)->tuple[pd.DataFrame,list[dict[str,Any]]]:
 out=[];states=[];months=pd.date_range(DEV_START,FINAL_START,freq='MS',inclusive='left')
 for si,side in enumerate(SIDES):
  sf=frame[frame.side.eq(side)]
  for fi,start in enumerate(months):
   end=start+pd.offsets.MonthBegin(1);tr=sf[train_mask(sf,start)];te=sf[sf.action_decision_ts.ge(start)&sf.action_decision_ts.lt(end)]
   if len(tr)<MIN_TRAIN or te.empty:continue
   state=fit_preprocess(tr,features,tr.delta_continue_bps.to_numpy(),SEED+si*100+fi)
   if not state['selected']:raise ValueError(f'{arm}/{side}/{start:%Y-%m}: no usable features')
   raw=fit_model(apply_preprocess(tr,state),tr.delta_continue_bps,SEED+si*100+fi).predict(apply_preprocess(te,state))
   z=te[['candidate_id','source_symbol','side','action_decision_ts','time_to_clear_bucket','volatility_bucket','net_exit_now_gross_bps','net_exit_now_cost_bps','net_exit_now_bps','net_continue_gross_bps','net_continue_cost_bps','net_continue_bps','delta_continue_bps','continue_better']].copy();z['arm']=arm;z['split']='development_oof';z['raw_predicted_delta_bps']=raw;out.append(z)
   states.append({'arm':arm,'side':side,'fold':start.strftime('%Y-%m'),'train_rows':len(tr),'test_rows':len(te),'train_max_label_available_ts':str(tr.label_available_ts.max()),'heldout_start':str(start),'preprocessing':state})
 return pd.concat(out,ignore_index=True),states

def final_score(frame:pd.DataFrame,features:list[str],arm:str,dev:pd.DataFrame)->tuple[pd.DataFrame,list[dict[str,Any]]]:
 out=[];frozen=[]
 for si,side in enumerate(SIDES):
  sf=frame[frame.side.eq(side)];tr=sf[train_mask(sf,FINAL_START)];te=sf[sf.action_decision_ts.ge(FINAL_START)&sf.action_decision_ts.lt(FINAL_END)];d=dev[dev.side.eq(side)]
  state=fit_preprocess(tr,features,tr.delta_continue_bps.to_numpy(),SEED+1000+si);cal=fit_calibration(d)
  raw=fit_model(apply_preprocess(tr,state),tr.delta_continue_bps,SEED+1000+si).predict(apply_preprocess(te,state));mapped,prob=calibrate(raw,cal)
  z=te[['candidate_id','source_symbol','side','action_decision_ts','time_to_clear_bucket','volatility_bucket','net_exit_now_gross_bps','net_exit_now_cost_bps','net_exit_now_bps','net_continue_gross_bps','net_continue_cost_bps','net_continue_bps','delta_continue_bps','continue_better']].copy();z['arm']=arm;z['split']='final_oos';z['raw_predicted_delta_bps']=raw;z['predicted_delta_continue_bps']=mapped;z['predicted_continue_probability']=prob;out.append(z)
  frozen.append({'arm':arm,'side':side,'freeze_ts':str(FINAL_START),'train_rows':len(tr),'development_calibration_rows':len(d),'selected_features':state['selected'],'preprocessing':state,'calibration':cal})
 return pd.concat(out,ignore_index=True),frozen

def replay(pred:pd.DataFrame,margin:float)->pd.DataFrame:
 z=pred.copy();z['action_threshold_bps']=float(margin);z['action']=np.where(z.predicted_delta_continue_bps.gt(margin),'CONTINUE_FROZEN_POLICY','EXIT_NOW')
 for kind in ('gross','cost','net'):z[f'policy_{kind}_bps']=np.where(z.action.eq('CONTINUE_FROZEN_POLICY'),z[f'net_continue_{kind}_bps'] if kind!='net' else z.net_continue_bps,z[f'net_exit_now_{kind}_bps'] if kind!='net' else z.net_exit_now_bps)
 z['incremental_vs_always_continue_bps']=z.policy_net_bps-z.net_continue_bps;z['incremental_vs_always_exit_bps']=z.policy_net_bps-z.net_exit_now_bps
 z['loss_avoided_correct_exit_bps']=np.where(z.action.eq('EXIT_NOW'),(-z.delta_continue_bps).clip(lower=0),0.);z['false_exit_opportunity_cost_bps']=np.where(z.action.eq('EXIT_NOW'),z.delta_continue_bps.clip(lower=0),0.)
 return z

def choose_margin(dev:pd.DataFrame)->tuple[float,pd.DataFrame]:
 rows=[]
 for m in MARGINS:
  z=replay(dev,m);rows.append({'margin_bps':m,'rows':len(z),'policy_net_mean_bps':float(z.policy_net_bps.mean()),'incremental_vs_continue_bps':float(z.incremental_vs_always_continue_bps.mean()),'incremental_vs_exit_bps':float(z.incremental_vs_always_exit_bps.mean()),'continue_rate':float(z.action.eq('CONTINUE_FROZEN_POLICY').mean())})
 evidence=pd.DataFrame(rows).sort_values(['incremental_vs_continue_bps','margin_bps'],ascending=[False,True],kind='stable');return float(evidence.iloc[0].margin_bps),evidence

def diagnostics(z:pd.DataFrame)->dict[str,Any]:
 y=z.delta_continue_bps.to_numpy();p=z.predicted_delta_continue_bps.to_numpy();pr=np.clip(z.predicted_continue_probability,1e-6,1-1e-6);b=z.continue_better.to_numpy();slope,intercept=np.polyfit(p,y,1) if len(z)>2 and np.std(p)>0 else (np.nan,np.nan);two=len(np.unique(b))>1
 err=np.abs(y-p);huber=np.where(err<=100,.5*err**2,100*(err-50))
 return {'rows':len(z),'mae_bps':float(mean_absolute_error(y,p)),'huber_loss':float(huber.mean()),'spearman_ic':float(spearmanr(y,p).statistic),'roc_auc':float(roc_auc_score(b,pr)) if two else np.nan,'pr_auc':float(average_precision_score(b,pr)) if two else np.nan,'brier':float(brier_score_loss(b,pr)),'log_loss':float(log_loss(b,pr,labels=[0,1])),'calibration_slope':float(slope),'calibration_intercept_bps':float(intercept),'continue_rate':float(z.action.eq('CONTINUE_FROZEN_POLICY').mean()),'exit_rate':float(z.action.eq('EXIT_NOW').mean()),'gross_policy_bps':float(z.policy_gross_bps.mean()),'cost_policy_bps':float(z.policy_cost_bps.mean()),'net_policy_bps':float(z.policy_net_bps.mean()),'net_always_continue_bps':float(z.net_continue_bps.mean()),'net_always_exit_bps':float(z.net_exit_now_bps.mean()),'incremental_vs_continue_bps':float(z.incremental_vs_always_continue_bps.mean()),'incremental_vs_exit_bps':float(z.incremental_vs_always_exit_bps.mean()),'giveback_cases_exited_pct':float(z.loc[z.delta_continue_bps.lt(0),'action'].eq('EXIT_NOW').mean()),'retained_cases_incorrectly_exited_pct':float(z.loc[z.delta_continue_bps.gt(0),'action'].eq('EXIT_NOW').mean()),'loss_avoided_bps':float(z.loss_avoided_correct_exit_bps.mean()),'false_exit_opportunity_cost_bps':float(z.false_exit_opportunity_cost_bps.mean()),'symbol_breadth':int(z.source_symbol.nunique()),'max_symbol_row_share':float(z.source_symbol.value_counts(normalize=True).max())}

def result_table(replays:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for (arm,split,margin),z in replays.groupby(['arm','split','action_threshold_bps']):
  month=z.action_decision_ts.dt.strftime('%Y-%m');parts=[('overall','ALL',z),*[( 'side',str(k),q) for k,q in z.groupby('side')],*[( 'month',str(k),q) for k,q in z.assign(month=month).groupby('month')],*[( 'symbol',str(k),q) for k,q in z.groupby('source_symbol')],*[( 'time_to_clear',str(k),q) for k,q in z.groupby('time_to_clear_bucket')],*[( 'volatility_bucket',str(k),q) for k,q in z.groupby('volatility_bucket')]]
  latest=month.max();monthly=z.assign(month=month).groupby('month').incremental_vs_always_continue_bps.mean();worst=str(monthly.idxmin());parts.extend([('latest_period',latest,z[month.eq(latest)]),('worst_month',worst,z[month.eq(worst)])])
  for dim,val,q in parts:r={'arm':arm,'split':split,'margin_bps':margin,'dimension':dim,'value':val};r.update(diagnostics(q));rows.append(r)
 return pd.DataFrame(rows)

def calibration_table(predictions:pd.DataFrame)->pd.DataFrame:
 parts=[]
 for split,z in predictions.groupby('split'):
  q=pd.qcut(z.predicted_delta_continue_bps.rank(method='first'),10,labels=False)+1
  t=z.assign(predicted_delta_decile=q).groupby('predicted_delta_decile').agg(rows=('candidate_id','size'),predicted_delta_bps=('predicted_delta_continue_bps','mean'),realised_delta_bps=('delta_continue_bps','mean'),realised_continue_better_rate=('continue_better','mean')).reset_index();t['split']=split;parts.append(t)
 return pd.concat(parts,ignore_index=True)

def bootstrap(z:pd.DataFrame)->dict[str,Any]:
 day=z.assign(day=z.action_decision_ts.dt.floor('D')).groupby('day').agg(continue_delta=('incremental_vs_always_continue_bps','sum'),exit_delta=('incremental_vs_always_exit_bps','sum'),rows=('candidate_id','size'));rng=np.random.default_rng(SEED);vals=[]
 for _ in range(BOOTSTRAPS):
  ix=rng.integers(0,len(day),len(day));den=day.rows.to_numpy()[ix].sum();vals.append([day.continue_delta.to_numpy()[ix].sum()/den,day.exit_delta.to_numpy()[ix].sum()/den])
 a=np.asarray(vals);return {'reps':BOOTSTRAPS,'seed':SEED,'versus_always_continue':{'ci_95_bps':[float(np.quantile(a[:,0],.025)),float(np.quantile(a[:,0],.975))],'prob_positive':float(np.mean(a[:,0]>0))},'versus_always_exit':{'ci_95_bps':[float(np.quantile(a[:,1],.025)),float(np.quantile(a[:,1],.975))],'prob_positive':float(np.mean(a[:,1]>0))}}

def run(output:Path)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 d2m,selection=require_d2();frame,groups=load_frame();d2_approved=list(selection['approved']);initial_groups=['A0_minimal_action_state_control',*[GROUP_MAP[g] for g in d2_approved]];initial_features=list(dict.fromkeys(c for g in initial_groups for c in groups[g]));states=[]
 full_dev,st=fold_score(frame,initial_features,'full_d2_approved');states+=st;full_dev=calibrate_development(full_dev);full_margin,full_margin_evidence=choose_margin(full_dev);full_dev_policy=replay(full_dev,full_margin);full_diag=diagnostics(full_dev_policy)
 candidates=[];loo_dev_predictions=[]
 for omitted in initial_groups[1:]:
  kept=[g for g in initial_groups if g!=omitted];f=list(dict.fromkeys(c for g in kept for c in groups[g]));d,st=fold_score(frame,f,f'leave_out__{omitted}');states+=st
  if set(d.candidate_id)!=set(full_dev.candidate_id):raise ValueError(f'leave-one-group-out row mismatch: {omitted}')
  d=calibrate_development(d);margin,margin_evidence=choose_margin(d);policy=replay(d,margin);diag=diagnostics(policy)
  calibration_error=lambda q:abs(q['calibration_slope']-1.)+abs(q['calibration_intercept_bps'])/100.
  qualifies=compact_readmission_decision(full_diag,diag)
  candidates.append({'omitted_group':omitted,'development_margin_bps':margin,'policy_net_bps':diag['net_policy_bps'],'full_policy_net_bps':full_diag['net_policy_bps'],'mae_bps':diag['mae_bps'],'full_mae_bps':full_diag['mae_bps'],'spearman_ic':diag['spearman_ic'],'full_spearman_ic':full_diag['spearman_ic'],'calibration_error':calibration_error(diag),'full_calibration_error':calibration_error(full_diag),'drop_group':qualifies,'rule':'drop only if development OOF policy net, MAE, IC improve and calibration is preserved'});loo_dev_predictions.append((omitted,kept,f,d,margin,margin_evidence))
 drops=[c for c in candidates if c['drop_group']]
 if drops:
  chosen=max(drops,key=lambda c:c['policy_net_bps']-c['full_policy_net_bps']);selected_groups=[g for g in initial_groups if g!=chosen['omitted_group']];chosen_tuple=next(x for x in loo_dev_predictions if x[0]==chosen['omitted_group']);dev=chosen_tuple[3];winner=chosen_tuple[4];evidence=chosen_tuple[5];readmission='DROPPED_'+chosen['omitted_group']
 else:selected_groups=initial_groups;dev=full_dev;winner=full_margin;evidence=full_margin_evidence;readmission='KEPT_ALL_D2_APPROVED_GROUPS'
 features=list(dict.fromkeys(c for g in selected_groups for c in groups[g]));final,frozen=final_score(frame,features,'compact_readmitted',dev);allpred=pd.concat([dev.assign(arm='compact_readmitted'),final],ignore_index=True);replays=pd.concat([replay(allpred,m) for m in MARGINS],ignore_index=True);replays['selected_margin_from_development']=replays.action_threshold_bps.eq(winner)
 # Evaluate both the inherited D2-approved full model and every development
 # leave-out on the exact same development and final rows. Final results are
 # descriptive only and never alter the re-admission decision above.
 comparison=[];loo_frozen=[];full_final,full_freeze=final_score(frame,initial_features,'full_d2_approved',full_dev);loo_frozen+=full_freeze;comparison.extend([replay(full_dev,full_margin),replay(full_final,full_margin)])
 for omitted,kept,f,d,margin,_ in loo_dev_predictions:
  fin,finfreeze=final_score(frame,f,f'leave_out__{omitted}',d);loo_frozen+=finfreeze
  if set(fin.candidate_id)!=set(full_final.candidate_id):raise ValueError(f'final leave-one-group-out row mismatch: {omitted}')
  comparison.extend([replay(d,margin),replay(fin,margin)])
 loo_pred=pd.concat(comparison,ignore_index=True);loo_results=result_table(loo_pred);loo_results['comparison_role']=np.where(loo_results.arm.eq('full_d2_approved'),'full_d2_approved','leave_group_out')
 for split in loo_results.split.unique():
  mask=loo_results.split.eq(split);left=idhash(sorted(loo_pred.loc[loo_pred.split.eq(split),'candidate_id'].astype(str).unique()));right=idhash(sorted(allpred.loc[allpred.split.eq(split),'candidate_id'].astype(str).unique()));loo_results.loc[mask,'candidate_id_set_sha256']=left;loo_results.loc[mask,'full_compact_candidate_id_set_sha256']=right
 loo_results['identical_rows_to_full_compact']=loo_results.candidate_id_set_sha256.eq(loo_results.full_compact_candidate_id_set_sha256)
 results=result_table(replays);calibration_deciles=calibration_table(allpred);selected_final=replays[(replays.split.eq('final_oos'))&replays.action_threshold_bps.eq(winner)];boot=bootstrap(selected_final)
 final_results=results[(results.split.eq('final_oos'))&results.margin_bps.eq(winner)];overall=final_results[final_results.dimension.eq('overall')].iloc[0];side_rows=final_results[final_results.dimension.eq('side')];latest=final_results[final_results.dimension.eq('latest_period')].iloc[0]
 calibration_rows=pd.concat([final_results[final_results.dimension.eq('overall')],side_rows]);calibration_ok=bool(calibration_rows.calibration_slope.between(.5,1.5).all() and calibration_rows.calibration_intercept_bps.abs().le(75.).all());symbol_effect=selected_final.groupby('source_symbol').incremental_vs_always_continue_bps.sum();symbol_concentration=float(symbol_effect.abs().max()/max(symbol_effect.abs().sum(),1e-12));symbol_support=int(selected_final.source_symbol.nunique())
 gate_contract={'calibration_slope_range':[.5,1.5],'calibration_intercept_abs_max_bps':75.,'action_rate_each_min':.02,'side_uplift_floor_bps':0.,'latest_uplift_floor_bps':0.,'max_absolute_symbol_uplift_concentration':.35,'minimum_symbol_support':10};lineage=lineage_evidence(frame,selected_groups,features)
 gates={'positive_paired_uplift_vs_always_continue':bool(overall.incremental_vs_continue_bps>0),'positive_paired_uplift_vs_always_exit':bool(overall.incremental_vs_exit_bps>0),'non_negative_latest_period_uplift':bool(latest.incremental_vs_continue_bps>=0),'no_material_failing_side':bool((side_rows.incremental_vs_continue_bps>=0).all()),'stable_calibration_incremental_bps':calibration_ok,'sufficient_action_support':bool(0.02<=overall.continue_rate<=0.98),'sufficient_symbol_breadth_and_no_single_symbol_concentration':bool(symbol_support>=10 and symbol_concentration<=.35),'no_causal_or_lineage_violation':bool(lineage['passed'])}
 terminal='CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES' if all(gates.values()) else 'CLEAR_EVENT_ACTION_SIGNAL_DIAGNOSTIC_ONLY'
 stage=Path(tempfile.mkdtemp(prefix=f'.{output.name}.',dir=output.parent))
 try:
  feature_manifest={'schema':'stage_d_compact_feature_manifest_v9','source':'canonical_D2_v4_plus_compact_development_LOO_readmission','d2_approved_groups':d2_approved,'compact_readmission':readmission,'compact_readmission_evidence':candidates,'included_groups_after_readmission':selected_groups,'requested_feature_count':len(features),'controlled_feature_cap_per_side':MAX_FEATURES_PER_SIDE,'frozen_before_final_oos':frozen,'leave_group_out_frozen_before_final_oos':loo_frozen,'training_only_preprocessing':states,'selection_timestamp_boundary':str(FINAL_START),'lineage_evidence':lineage};dump(stage/'stage_d_compact_feature_manifest.json',feature_manifest)
  loo_results.to_parquet(stage/'stage_d_leave_group_out_results.parquet',index=False,compression='zstd');replays.to_parquet(stage/'stage_d_action_policy_replay.parquet',index=False,compression='zstd');results.to_parquet(stage/'stage_d_compact_model_results.parquet',index=False,compression='zstd');calibration_deciles.to_parquet(stage/'stage_d_compact_calibration.parquet',index=False,compression='zstd');evidence.to_parquet(stage/'stage_d_margin_development_selection.parquet',index=False,compression='zstd');dump(stage/'stage_d_action_replay_bootstrap.json',boot)
  dump(stage/'stage_d_action_research_gate.json',{'terminal_decision':terminal,'gates':gates,'gate_contract':gate_contract,'lineage_evidence':lineage,'symbol_stability':{'symbol_support':symbol_support,'max_absolute_symbol_uplift_concentration':symbol_concentration},'selected_margin_bps':winner,'selection_source':'development_oof_only_after_compact_readmission','final_oos_descriptive_only':True})
  (stage/'stage_d_compact_action_report.md').write_text(f"# Stage-D D3/D4 compact action model\n\nD2-v4 development-approved groups: {d2_approved}. Compact development-only re-admission: {readmission}; included groups: {selected_groups}. Margin {winner:g} bps selected strictly on development OOF. Final OOS did not alter groups or margin. Final gates: {gates}. Terminal decision: {terminal}. Paired day-block bootstrap: {boot}.\n")
  outputs={p.name:sha(p) for p in stage.iterdir()};manifest={'schema':SCHEMA,'status':'RESEARCH_ONLY_NO_ENTRY_OR_PORTFOLIO_POLICY_CHANGE','source_population_rows':len(frame),'development_oof_rows':len(dev),'final_oos_rows':len(final),'candidate_id_sha256':idhash(frame.candidate_id),'d2_v4_manifest_sha256':sha(D2_MANIFEST),'inputs':{str(p):sha(p) for p in [D2_MANIFEST,D2_SELECTION,FEATURES,GROUPS,TARGETS,FEATURE_RUN_MANIFEST,FEATURE_LINEAGE]},'d2_approved_groups':d2_approved,'compact_readmission':readmission,'included_groups_after_readmission':selected_groups,'development_selected_margin_bps':winner,'bootstrap':boot,'research_gate':gates,'lineage_evidence':lineage,'gate_contract':gate_contract,'terminal_decision':terminal,'outputs_sha256':outputs,'runner_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests/test_stage_d_compact_action_model.py')};dump(stage/'run_manifest.json',manifest);(stage/'manifest.sha256').write_text(f"{sha(stage/'run_manifest.json')}  run_manifest.json\n");os.replace(stage,output);return manifest
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();print(json.dumps(run(a.output),indent=2,default=str))
