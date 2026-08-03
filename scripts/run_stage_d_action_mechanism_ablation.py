#!/usr/bin/env python3
"""Strict chronological Stage-D D0--D9 action mechanism ablation."""
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
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
ART=ROOT/'data_perp/artifacts'
FEATURE_DIR=ART/'stage_d_action_features_20260731_v3'
FEATURES=FEATURE_DIR/'stage_d_action_features.parquet'
GROUPS=FEATURE_DIR/'stage_d_action_feature_groups.json'
TARGETS=ART/'stage_d_action_counterfactuals_20260731_v2/stage_d_action_counterfactuals.parquet'
DEFAULT_OUTPUT=ART/'stage_d_action_mechanism_ablation_20260731_v2'
DEV_MONTHS=pd.date_range('2024-04-01','2024-07-01',freq='MS',tz='UTC')
FINAL_START=pd.Timestamp('2024-08-01T00:00:00Z'); END=pd.Timestamp('2024-12-01T00:00:00Z')
HORIZON=pd.Timedelta(hours=12); SIDES=('long','short'); SEED=20260731
MIN_TRAIN=500; MAX_FEATURES=48; TREES=120; BOOTSTRAPS=500
BLOCKED={'D3':'A3_REJECTED_SOURCE_UNAVAILABLE','D6':'A6_REJECTED_LINEAGE','D7':'A7_REJECTED_LINEAGE','D8':'A8_REJECTED_OOF_LINEAGE'}
ACTION_THRESHOLD_BPS=0.0

def sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def idhash(x:Iterable[Any])->str:return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def dump(path:Path,x:Any)->None:path.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def train_mask(f:pd.DataFrame,start:pd.Timestamp)->pd.Series:
 return f.action_decision_ts.lt(start-HORIZON)&f.label_available_ts.lt(start)
def action_from_prediction(prediction:np.ndarray,threshold_bps:float=ACTION_THRESHOLD_BPS)->np.ndarray:
 return np.where(np.asarray(prediction)>threshold_bps,'CONTINUE_FROZEN_POLICY','EXIT_NOW')

def load(smoke:bool=False)->tuple[pd.DataFrame,dict[str,list[str]]]:
 x=pd.read_parquet(FEATURES); y=pd.read_parquet(TARGETS)
 keep=['candidate_id','clear_event_bar_open_ts','first_clear_bar_index','entry_executable_price','net_exit_now_gross_bps','net_exit_now_cost_bps','net_exit_now_bps','net_continue_gross_bps','net_continue_cost_bps','net_continue_bps','delta_continue_bps','continue_better']
 f=x.merge(y[keep],on='candidate_id',how='inner',validate='one_to_one')
 for c in ['action_decision_ts','label_available_ts','entry_ts']:f[c]=pd.to_datetime(f[c],utc=True)
 f=f.sort_values(['action_decision_ts','candidate_id'],kind='stable').reset_index(drop=True)
 if len(f)!=len(x) or f.candidate_id.duplicated().any():raise ValueError('fixed action population mismatch')
 if not np.allclose(f.net_continue_bps-f.net_exit_now_bps,f.delta_continue_bps):raise ValueError('target arithmetic drift')
 if smoke:
  f['__h']=pd.util.hash_pandas_object(f.candidate_id,index=False)
  f=(f.assign(month=f.action_decision_ts.dt.strftime('%Y-%m')).sort_values(['month','side','__h']).groupby(['month','side'],group_keys=False).head(250).drop(columns=['month','__h']).sort_values(['action_decision_ts','candidate_id']).reset_index(drop=True))
 f['volatility_bucket']=pd.qcut(f.realised_volatility.rank(method='first'),4,labels=['Q1_low','Q2','Q3','Q4_high'])
 raw=json.loads(GROUPS.read_text()); groups={k:list(v) for k,v in raw.items() if k.startswith('A') and isinstance(v,list)}
 return f,groups

def preprocess(train:pd.DataFrame,test:pd.DataFrame,requested:list[str],y:np.ndarray,seed:int,cap:int)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
 requested=list(dict.fromkeys(requested)); usable=[]; state={'requested':requested,'removed':{},'clip':{},'median':{},'mean':{},'std':{},'selected':[]}
 for c in requested:
  if c not in train:state['removed'][c]='absent';continue
  s=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan); nn=s.dropna()
  if s.notna().mean()<.5:state['removed'][c]='availability_below_0.5';continue
  if nn.nunique()<2 or nn.value_counts(normalize=True).iloc[0]>=.995:state['removed'][c]='near_constant';continue
  lo,hi=nn.quantile([.01,.99]); state['clip'][c]=[float(lo),float(hi)]; state['median'][c]=float(nn.median());usable.append(c)
 def base(z:pd.DataFrame)->pd.DataFrame:
  return pd.DataFrame({c:pd.to_numeric(z[c],errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan).clip(*state['clip'][c]).fillna(state['median'][c]) for c in usable},index=z.index)
 a,b=base(train),base(test)
 reps=[]
 corr=a.corr(method='spearman').abs()
 for c in usable:
  if all(float(corr.loc[c,q])<.96 for q in reps):reps.append(c)
 selected=reps
 if len(reps)>cap:
  selector=lgb.LGBMRegressor(objective='huber',n_estimators=60,num_leaves=31,learning_rate=.05,max_depth=-1,min_child_samples=80,subsample=.9,colsample_bytree=.9,reg_lambda=2.,random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)
  selector.fit(a[reps],y); gain=pd.Series(selector.booster_.feature_importance('gain'),index=reps)
  selected=gain.sort_values(ascending=False,kind='stable').head(cap).index.tolist();state['gain']={k:float(gain[k]) for k in selected}
 for c in selected:
  mu=float(a[c].mean()); sd=float(a[c].std(ddof=0)); sd=sd if sd>1e-12 else 1.;state['mean'][c]=mu;state['std'][c]=sd
 state['selected']=selected
 return pd.DataFrame({c:(a[c]-state['mean'][c])/state['std'][c] for c in selected},index=a.index),pd.DataFrame({c:(b[c]-state['mean'][c])/state['std'][c] for c in selected},index=b.index),state

def hierarchical_preprocess(train:pd.DataFrame,test:pd.DataFrame,group_names:list[str],groups:dict[str,list[str]],y:np.ndarray,seed:int)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
 xs=[];vs=[];state={'hierarchical':True,'groups':{},'selected':[]}
 group_offsets={'A0_minimal_action_state_control':0,'A1_path_geometry_to_clear':100,'A2_candle_rejection_structure':200,'A4_volatility_instability_to_clear':400,'A5_market_cross_sectional_confirmation':500,'A9_compact_composites':900}
 for name in group_names:
  cap=MAX_FEATURES if name.startswith('A0_') else 16
  a,b,s=preprocess(train,test,groups[name],y,seed+group_offsets[name],cap)
  state['groups'][name]=s;state['selected']+=s['selected'];xs.append(a);vs.append(b)
 if len(state['selected'])!=len(set(state['selected'])):raise ValueError('hierarchical groups overlap after selection')
 return pd.concat(xs,axis=1),pd.concat(vs,axis=1),state

def model(seed:int)->lgb.LGBMRegressor:
 return lgb.LGBMRegressor(objective='huber',alpha=.9,n_estimators=TREES,num_leaves=31,learning_rate=.04,max_depth=-1,min_child_samples=80,subsample=.9,colsample_bytree=.9,reg_lambda=2.,random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)

def calibration(history:pd.DataFrame,raw:np.ndarray,train_scale:float)->tuple[np.ndarray,np.ndarray,dict[str,Any]]:
 h=history.dropna(subset=['raw_predicted_delta_bps','delta_continue_bps']) if {'raw_predicted_delta_bps','delta_continue_bps'}<=set(history.columns) else pd.DataFrame(columns=['raw_predicted_delta_bps','delta_continue_bps','continue_better'])
 if len(h)>=250 and h.raw_predicted_delta_bps.std()>1e-9:
  slope,intercept=np.polyfit(h.raw_predicted_delta_bps,h.delta_continue_bps,1); source='prior_chronological_oof'
  if h.continue_better.nunique()>1:
   clf=LogisticRegression(C=10.,max_iter=500).fit(h[['raw_predicted_delta_bps']],h.continue_better)
   prob=clf.predict_proba(pd.DataFrame({'raw_predicted_delta_bps':np.asarray(raw)}))[:,1]; pcoef=float(clf.coef_[0,0]);pint=float(clf.intercept_[0])
  else:prob=1/(1+np.exp(-np.asarray(raw)/train_scale));pcoef=np.nan;pint=np.nan
 else:
  slope,intercept=1.,0.; source='identity_bps_insufficient_prior_oof';prob=1/(1+np.exp(-np.asarray(raw)/train_scale));pcoef=np.nan;pint=np.nan
 return np.asarray(raw)*slope+intercept,prob,{'source':source,'rows':len(h),'slope':float(slope),'intercept':float(intercept),'probability_coef':pcoef,'probability_intercept':pint,'fallback_scale_bps':train_scale}

def arm_features(g:dict[str,list[str]])->dict[str,list[str]]:
 a0='A0_minimal_action_state_control';a1='A1_path_geometry_to_clear';a2='A2_candle_rejection_structure';a4='A4_volatility_instability_to_clear';a5='A5_market_cross_sectional_confirmation';a9='A9_compact_composites'
 return {'D0':[a0],'D1':[a0,a1],'D2':[a0,a1,a2],'D4':[a0,a1,a2,a4],'D5':[a0,a1,a2,a4,a5],'M_A2':[a0,a2],'M_A4':[a0,a4],'M_A5':[a0,a5],'M_A9':[a0,a9]}

def score_arm(frame:pd.DataFrame,arm:str,group_names:list[str],groups:dict[str,list[str]],folds:list[tuple[str,pd.Timestamp,pd.Timestamp,str]],history_source:dict[tuple[str,str],pd.DataFrame],states:list[dict[str,Any]])->pd.DataFrame:
 out=[]
 for side_i,side in enumerate(SIDES):
  sf=frame[frame.side.eq(side)]
  prior=pd.DataFrame()
  for fi,(fold,start,end,split) in enumerate(folds):
   train=sf[train_mask(sf,start)]; test=sf[sf.action_decision_ts.ge(start)&sf.action_decision_ts.lt(end)]
   if len(train)<MIN_TRAIN or test.empty:continue
   xt,xv,state=hierarchical_preprocess(train,test,group_names,groups,train.delta_continue_bps.to_numpy(),SEED+side_i*1000+fi)
   if not state['selected']:raise ValueError(f'{arm}/{side}/{fold} no selected features')
   fit=model(SEED+side_i*1000+fi);fit.fit(xt,train.delta_continue_bps);raw=fit.predict(xv)
   hist=prior if split=='development_oof' else history_source.get((arm,side),pd.DataFrame())
   mapped,prob,cal=calibration(hist,raw,max(float(train.delta_continue_bps.std()),1.))
   z=test[['candidate_id','source_symbol','side','entry_ts','action_decision_ts','label_available_ts','time_to_clear_minutes','realised_volatility','volatility_bucket','net_exit_now_gross_bps','net_exit_now_cost_bps','net_exit_now_bps','net_continue_gross_bps','net_continue_cost_bps','net_continue_bps','delta_continue_bps','continue_better']].copy()
   z['arm']=arm;z['fold']=fold;z['split']=split;z['raw_predicted_delta_bps']=raw;z['predicted_delta_continue_bps']=mapped;z['predicted_continue_probability']=prob;z['action']=action_from_prediction(mapped)
   z['policy_net_bps']=np.where(z.action.eq('CONTINUE_FROZEN_POLICY'),z.net_continue_bps,z.net_exit_now_bps)
   z['policy_gross_bps']=np.where(z.action.eq('CONTINUE_FROZEN_POLICY'),z.net_continue_gross_bps,z.net_exit_now_gross_bps)
   z['policy_cost_bps']=np.where(z.action.eq('CONTINUE_FROZEN_POLICY'),z.net_continue_cost_bps,z.net_exit_now_cost_bps)
   z['incremental_vs_always_continue_bps']=z.policy_net_bps-z.net_continue_bps;z['incremental_vs_always_exit_bps']=z.policy_net_bps-z.net_exit_now_bps
   z['month']=z.action_decision_ts.dt.strftime('%Y-%m');out.append(z)
   if split=='development_oof':prior=pd.concat([prior,z[['raw_predicted_delta_bps','delta_continue_bps','continue_better']]],ignore_index=True)
   states.append({'arm':arm,'side':side,'fold':fold,'split':split,'train_rows':len(train),'test_rows':len(test),'train_max_action_ts':str(train.action_decision_ts.max()),'train_max_label_available_ts':str(train.label_available_ts.max()),'heldout_start':str(start),'purge_hours':12,'seed':SEED+side_i*1000+fi,'preprocessing':state,'calibration':cal})
 return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def diagnostics(x:pd.DataFrame)->dict[str,Any]:
 y=x.delta_continue_bps.to_numpy();p=x.predicted_delta_continue_bps.to_numpy();prob=np.clip(x.predicted_continue_probability.to_numpy(),1e-6,1-1e-6);binary=x.continue_better.to_numpy()
 slope,intercept=(np.polyfit(p,y,1).tolist() if len(x)>2 and np.std(p)>0 else [np.nan,np.nan])
 continue_rate=float(x.action.eq('CONTINUE_FROZEN_POLICY').mean())
 return {'mae_bps':float(mean_absolute_error(y,p)),'huber_loss':float(np.mean(np.where(np.abs(y-p)<=100,.5*(y-p)**2,100*(np.abs(y-p)-50)))),'spearman_ic':float(spearmanr(y,p).statistic),'roc_auc':float(roc_auc_score(binary,prob)) if len(np.unique(binary))>1 else np.nan,'pr_auc':float(average_precision_score(binary,prob)) if len(np.unique(binary))>1 else np.nan,'brier':float(brier_score_loss(binary,prob)),'log_loss':float(log_loss(binary,prob,labels=[0,1])),'calibration_slope_bps':float(slope),'calibration_intercept_bps':float(intercept),'continue_rate':continue_rate,'exit_rate':1.-continue_rate,'net_always_continue_bps':float(x.net_continue_bps.mean()),'net_always_exit_bps':float(x.net_exit_now_bps.mean()),'net_learned_policy_bps':float(x.policy_net_bps.mean()),'incremental_vs_continue_bps':float(x.incremental_vs_always_continue_bps.mean()),'incremental_vs_exit_bps':float(x.incremental_vs_always_exit_bps.mean()),'gross_policy_bps':float(x.policy_gross_bps.mean()),'cost_policy_bps':float(x.policy_cost_bps.mean()),'giveback_cases_exited_pct':float(x.loc[x.delta_continue_bps.lt(0),'action'].eq('EXIT_NOW').mean()),'retained_cases_incorrectly_exited_pct':float(x.loc[x.delta_continue_bps.gt(0),'action'].eq('EXIT_NOW').mean()),'opportunity_cost_false_exits_bps':float((-x.loc[(x.delta_continue_bps.gt(0))&x.action.eq('EXIT_NOW'),'delta_continue_bps']).sum()/max(len(x),1)),'loss_avoided_correct_exits_bps':float((-x.loc[(x.delta_continue_bps.lt(0))&x.action.eq('EXIT_NOW'),'delta_continue_bps']).sum()/max(len(x),1)),'symbol_breadth':int(x.source_symbol.nunique()),'max_symbol_row_share':float(x.source_symbol.value_counts(normalize=True).max())}

def result_records(scored:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for (arm,split),z in scored.groupby(['arm','split']):
  slices=[('aggregate','all',z)]
  slices += [('side',str(k),q) for k,q in z.groupby('side')]+[('month',str(k),q) for k,q in z.groupby('month')]+[('symbol',str(k),q) for k,q in z.groupby('source_symbol')]
  slices += [('volatility_bucket',str(k),q) for k,q in z.groupby('volatility_bucket',observed=True)]
  tc=pd.cut(z.time_to_clear_minutes,[-np.inf,15,60,180,360,np.inf],labels=['0-15','16-60','61-180','181-360','361+']); slices += [('time_to_clear',str(k),z.loc[q.index]) for k,q in tc.groupby(tc,observed=True)]
  months=z.groupby('month').incremental_vs_always_continue_bps.mean();latest=str(sorted(months.index)[-1]);worst=str(months.idxmin());slices += [('latest_period',latest,z[z.month.eq(latest)]),('worst_month',worst,z[z.month.eq(worst)])]
  for dim,val,q in slices:
   r={'arm':arm,'split':split,'dimension':dim,'value':val,'rows':len(q)};r.update(diagnostics(q));rows.append(r)
 return pd.DataFrame(rows)

def calibration_table(scored:pd.DataFrame)->pd.DataFrame:
 parts=[]
 for (arm,split),z in scored.groupby(['arm','split']):
  q=pd.qcut(z.predicted_delta_continue_bps.rank(method='first'),10,labels=False)+1
  t=z.assign(predicted_delta_decile=q).groupby('predicted_delta_decile').agg(rows=('candidate_id','size'),predicted_delta_bps=('predicted_delta_continue_bps','mean'),realised_delta_bps=('delta_continue_bps','mean'),realised_continue_better_rate=('continue_better','mean'),model_continue_action_rate=('action',lambda x:float((x=='CONTINUE_FROZEN_POLICY').mean()))).reset_index();t['arm']=arm;t['split']=split;parts.append(t)
 return pd.concat(parts,ignore_index=True)

def bootstrap_draws(scored:pd.DataFrame)->dict[tuple[str,str],tuple[np.ndarray,np.ndarray]]:
 """Precompute common UTC-day support and paired resamples for every scope."""
 draws={}
 for split,whole in scored.groupby('split',sort=True):
  for side,z in [('all',whole),*[(str(s),q) for s,q in whole.groupby('side',sort=True)]]:
   days=np.sort(z.action_decision_ts.dt.floor('D').unique())
   # A scope-local seed makes draws invariant to arm count and iteration order.
   seed=int.from_bytes(hashlib.sha256(f'{SEED}|{split}|{side}'.encode()).digest()[:8],'little')
   rng=np.random.default_rng(seed)
   draws[(str(split),side)]=(days,rng.integers(0,len(days),size=(BOOTSTRAPS,len(days))))
 return draws

def bootstrap(scored:pd.DataFrame)->pd.DataFrame:
 rows=[];draws=bootstrap_draws(scored)
 for (arm,split),whole in scored.groupby(['arm','split']):
  for side,z in [('all',whole),*[(str(s),q) for s,q in whole.groupby('side')]]:
   support,sampled=draws[(str(split),side)]
   z=z.assign(day=z.action_decision_ts.dt.floor('D'));day=z.groupby('day').agg(c_sum=('incremental_vs_always_continue_bps','sum'),e_sum=('incremental_vs_always_exit_bps','sum'),rows=('candidate_id','size')).reindex(support,fill_value=0);vals=[]
   for draw in sampled:vals.append(day_block_estimate(day,draw))
   a=np.asarray(vals)
   for j,base in enumerate(['always_continue','always_exit']):rows.append({'arm':arm,'split':split,'side':side,'baseline':base,'replicates':BOOTSTRAPS,'mean_bps':float(a[:,j].mean()),'ci_low_bps':float(np.quantile(a[:,j],.025)),'ci_high_bps':float(np.quantile(a[:,j],.975)),'prob_positive':float(np.mean(a[:,j]>0))})
 return pd.DataFrame(rows)

def day_block_estimate(day:pd.DataFrame,sampled:np.ndarray)->list[float]:
 den=day.rows.to_numpy()[sampled].sum()
 return [float(day.c_sum.to_numpy()[sampled].sum()/den),float(day.e_sum.to_numpy()[sampled].sum()/den)]

def blocked_arms_identical(scored:pd.DataFrame)->bool:
 ignored={'arm','arm_status','blocked_reason'}
 for blocked,predecessor in {'D3':'D2','D6':'D5','D7':'D5','D8':'D5'}.items():
  a=scored[scored.arm.eq(blocked)].sort_values(['split','candidate_id']).reset_index(drop=True)
  b=scored[scored.arm.eq(predecessor)].sort_values(['split','candidate_id']).reset_index(drop=True)
  cols=sorted(set(a.columns)-ignored)
  if len(a)!=len(b) or not a[cols].equals(b[cols]):return False
 return True

def select_groups(dev:pd.DataFrame)->tuple[list[str],dict[str,Any]]:
 comparisons=[('A1','D1','D0'),('A2','D2','D1'),('A4','D4','D2'),('A5','D5','D4'),('A9','M_A9','D0')];approved=[];evidence={}
 for group,arm,base in comparisons:
  a=dev[dev.arm.eq(arm)].set_index('candidate_id');b=dev[dev.arm.eq(base)].set_index('candidate_id');ids=a.index.intersection(b.index);delta=a.loc[ids].policy_net_bps-b.loc[ids].policy_net_bps
  paired=a.loc[ids].assign(__d=delta);monthly=paired.groupby('month').__d.mean();side=paired.groupby('side').__d.mean();sym=paired.groupby('source_symbol').__d.sum();concentration=float(sym.abs().max()/max(sym.abs().sum(),1e-12))
  da=diagnostics(a.loc[ids]);db=diagnostics(b.loc[ids]);prediction_ok=(da['mae_bps']<db['mae_bps']) or (da['spearman_ic']>db['spearman_ic']);cal_a=abs(da['calibration_slope_bps']-1)+abs(da['calibration_intercept_bps'])/100;cal_b=abs(db['calibration_slope_bps']-1)+abs(db['calibration_intercept_bps'])/100;calibration_ok=cal_a<=cal_b*1.05
  ok=float(delta.mean())>0 and prediction_ok and calibration_ok and int((monthly>0).sum())>=2 and int((side>=0).sum())==2 and concentration<=.35 and paired.source_symbol.nunique()>=10
  evidence[group]={'mean_incremental_policy_bps':float(delta.mean()),'prediction_improves':prediction_ok,'calibration_preserved':calibration_ok,'positive_months':int((monthly>0).sum()),'side_bps':side.to_dict(),'symbol_concentration':concentration,'symbol_support':int(paired.source_symbol.nunique()),'approved':ok};
  if ok:approved.append(group)
 return approved,{'source':'development_oof_only','rules':'positive paired policy increment; prediction improves; calibration preserved within 5%; >=2 positive months; both sides nonnegative; symbol concentration <=0.35; >=10 symbols','groups':evidence,'approved':approved}

def run(output:Path,smoke:bool=False)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 frame,groups=load(smoke);folds=[]
 for s in DEV_MONTHS:folds.append((s.strftime('%Y-%m'),s,s+pd.offsets.MonthBegin(1),'development_oof'))
 final_folds=[('2024-08_to_11',FINAL_START,END,'final_oos')]
 arms=arm_features(groups);states=[];dev_parts=[]
 for arm,features in arms.items():
  print('[D2 dev]',arm,flush=True);dev_parts.append(score_arm(frame,arm,features,groups,folds,{},states))
 dev=pd.concat(dev_parts,ignore_index=True);approved,selection=select_groups(dev)
 d9=['A0_minimal_action_state_control']
 for name in approved:
  key={'A1':'A1_path_geometry_to_clear','A2':'A2_candle_rejection_structure','A4':'A4_volatility_instability_to_clear','A5':'A5_market_cross_sectional_confirmation','A9':'A9_compact_composites'}[name];d9.append(key)
 arms['D9']=list(dict.fromkeys(d9));print('[D2 dev] D9',approved,flush=True);dev9=score_arm(frame,'D9',arms['D9'],groups,folds,{},states);dev=pd.concat([dev,dev9],ignore_index=True)
 d9_states=[s for s in states if s['arm']=='D9' and s['split']=='development_oof'];survival={name:all(bool(s['preprocessing']['groups'].get(name,{}).get('selected')) for s in d9_states) for name in arms['D9']}
 selection['d9_group_survives_every_side_fold']=survival
 if not all(survival.values()):raise ValueError(f'D9 selected group lost all fields in a fold: {survival}')
 hist={(a,s):dev[(dev.arm.eq(a))&dev.side.eq(s)][['raw_predicted_delta_bps','delta_continue_bps','continue_better']] for a in arms for s in SIDES}
 final=[]
 for arm,features in arms.items():print('[D2 final]',arm,flush=True);final.append(score_arm(frame,arm,features,groups,final_folds,hist,states))
 scored=pd.concat([dev,*final],ignore_index=True)
 # Explicit blocked arms are byte/row-identical policy predictions to their predecessor.
 for blocked,predecessor in {'D3':'D2','D6':'D5','D7':'D5','D8':'D5'}.items():
  q=scored[scored.arm.eq(predecessor)].copy();q['arm']=blocked;q['arm_status']='NOT_RUN';q['blocked_reason']=BLOCKED[blocked];scored=pd.concat([scored,q],ignore_index=True)
 scored['arm_status']=scored.get('arm_status',pd.Series(index=scored.index,dtype=object)).fillna('RUN');scored['blocked_reason']=scored.get('blocked_reason',pd.Series(index=scored.index,dtype=object))
 results=result_records(scored);cal=calibration_table(scored);boot=bootstrap(scored)
 stability=results[results.dimension.isin(['side','month','symbol','time_to_clear','volatility_bucket','latest_period','worst_month'])].copy()
 stage=Path(tempfile.mkdtemp(prefix=f'.{output.name}.',dir=output.parent))
 try:
  scored.to_parquet(stage/'stage_d_action_oof_predictions.parquet',index=False,compression='zstd');results.to_parquet(stage/'stage_d_action_model_results.parquet',index=False,compression='zstd');cal.to_parquet(stage/'stage_d_action_calibration.parquet',index=False,compression='zstd');stability.to_parquet(stage/'stage_d_action_stability.parquet',index=False,compression='zstd');boot.to_parquet(stage/'stage_d_action_bootstrap.parquet',index=False,compression='zstd')
  dump(stage/'stage_d_fold_preprocessing_selection_manifest.json',states);dump(stage/'stage_d_d9_development_selection.json',selection);dump(stage/'stage_d_action_arm_manifest.json',{'arms':arms,'blocked':BLOCKED,'threshold_bps':0,'model':{'class':'LGBMRegressor','objective':'huber','trees':TREES,'HPO':False},'folds':[(a,str(b),str(c),d) for a,b,c,d in folds+final_folds]})
  dump(stage/'correctness_test_report.json',{'schema':'stage_d_d2_runtime_invariants_v2','invariants':{'unique_fixed_population':not frame.candidate_id.duplicated().any(),'feature_availability':bool(frame.feature_available_ts.le(frame.action_decision_ts).all()),'cost_once_continue':bool(np.allclose(frame.net_continue_gross_bps-frame.net_continue_cost_bps,frame.net_continue_bps)),'cost_once_exit':bool(np.allclose(frame.net_exit_now_gross_bps-frame.net_exit_now_cost_bps,frame.net_exit_now_bps)),'delta_exact':bool(np.allclose(frame.net_continue_bps-frame.net_exit_now_bps,frame.delta_continue_bps)),'threshold_fixed_zero':ACTION_THRESHOLD_BPS==0,'blocked_arms_identical':blocked_arms_identical(scored),'paired_bootstrap_draws_reused_by_scope':True,'oi_funding_regime_not_run':True,'volume_bucket_not_reported_A3_blocked':True},'focused_pytest':'tests/test_stage_d_action_mechanism_ablation.py'})
  (stage/'stage_d_action_mechanism_ablation_report.md').write_text(f"# Stage-D D2 mechanism ablation\n\nRows: {len(frame):,}. Development-approved groups for D9: {approved}. A3/A6/A7/A8 are explicit NOT_RUN identical predecessor arms. Threshold is 0 bps; no top-k rule.\n")
  outputs={p.name:sha(p) for p in stage.iterdir()};manifest={'schema':'stage_d_action_mechanism_ablation_v2','status':'RESEARCH_ONLY','supersedes':'stage_d_action_mechanism_ablation_20260731_v1','rows':len(frame),'candidate_id_set_sha256':idhash(sorted(frame.candidate_id.astype(str))),'inputs':{str(p):sha(p) for p in [FEATURES,GROUPS,TARGETS]},'code':{str(Path(__file__).resolve()):sha(Path(__file__))},'outputs_sha256':outputs,'blocked':BLOCKED,'selection':selection};dump(stage/'run_manifest.json',manifest);os.replace(stage,output);return manifest
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--smoke',action='store_true');a=p.parse_args();print(json.dumps(run(a.output,a.smoke),indent=2,default=str))
