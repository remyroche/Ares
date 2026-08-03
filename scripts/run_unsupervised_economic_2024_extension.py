#!/usr/bin/env python3
"""Fold-local 2024 extension of the matched unsupervised economic ablation.

This is deliberately a diagnostic historical backcast.  Its exact 1m economics
use the frozen-current-spread counterfactual and are not execution-parity or
promotion evidence.  Every representation is refit using candidates strictly
earlier than the evaluation month; precomputed GMM/DAE/posterior fields in the
raw shards are never read as inputs.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, validate_candidate_identity # noqa:E402
from extreme_price_movements.regime_stack_evaluation import EvaluationColumns, evaluate_matched_arms, global_top_k_mask # noqa:E402

SCHEMA='unsupervised_economic_2024_extension_v1'
OUT=ROOT/'data_perp/artifacts/unsupervised_economic_2024_extension_20260730_v2'
OOF=ROOT/'data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet'
RAW23=ROOT/'data_perp/reports/failure_2022_2023_pf_baseonly_backcast_20260730_v1/candidate_shards'
RAW24=ROOT/'data_perp/reports/failure_2024_transition_exact1m_candidate_backcast_20260730_v1/candidate_shards'
TRANS=ROOT/'data_perp/artifacts/reconstructed_2024_candidate_oof_regime_transition_20260730_v1/candidate_oof_regime_transition.parquet'
STAGE23=ROOT/'data_perp/artifacts/failure_2022_2023_pf_exact1m_request_stage_20260730_v1/staged_candidates.parquet'
STAGE24=ROOT/'data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2/staged_candidates.parquet'
TARGET='execution_net_ev_12h'; ALPHA='__reconstructed_soft_alpha_12h__'
RAW_FEATURES=['ema50_ema200_spread_atr','ema50_slope','trend_strength_percentile','realized_volatility_24h','atr_change_rate','true_range_percentile','rolling_range_20','atr_percentile','compression_ratio','range_expansion_ratio','volatility_of_volatility_48','efficiency_ratio_20','choppiness_index_20','direction_entropy_20','volume_zscore_48h','amihud_z','mkt_ret_eq_24h','market_breadth_4h','market_dispersion_4h','funding_z','oi_value_z_30d','mkt_oi_chg_z_24h','mkt_funding_mean_z_30d','mkt_funding_dispersion_z_30d','mkt_ret_1h','mkt_ret_4h','mkt_rv_24h','mkt_atr_expansion_4h','cross_asset_corr_4h','market_downside_pairwise_corr_24h','liquidation_onset_score','mkt_systemic_deleveraging_score','ret24h_bench_resid','rv_24h_peer_resid','asset_minus_mkt_oi_1d_ts_resid']
ARMS={'baseline':['score_residual_alpha'],'gmm_geometry':['score_residual_alpha','gmm_ood_score','mahalanobis_distance','expected_mahalanobis'],'dae_only':['score_residual_alpha',*[f'dae_fold_{i:02d}' for i in range(8)],'dae_reconstruction_error_zscore'],'gmm_plus_dae':['score_residual_alpha','gmm_ood_score','mahalanobis_distance','expected_mahalanobis',*[f'dae_fold_{i:02d}' for i in range(8)],'dae_reconstruction_error_zscore'],'failure_destination':['score_residual_alpha','p_failure_destination_12h'],'transition_only':['score_residual_alpha','p_transition_active_oof'],'failure_plus_transition':['score_residual_alpha','p_failure_destination_12h','p_transition_active_oof']}
RETAINED_DIAGNOSTICS=[*[f'dae_fold_{i:02d}' for i in range(8)],'dae_reconstruction_error_zscore','gmm_ood_score','mahalanobis_distance','expected_mahalanobis','p_failure_destination_12h','p_transition_active_oof']

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def _read_raw(root:Path, *, historical:bool=False, stage_path:Path|None=None)->pd.DataFrame:
 fs=sorted(root.glob('candidates_*.parquet'))
 if not fs: raise RegimeOOFStackError(f'no candidate shards under {root}')
 if not historical:
  return pd.concat([pd.read_parquet(p,columns=[*IDENTITY_COLUMNS,*RAW_FEATURES]) for p in fs],ignore_index=True)
 stage=pd.read_parquet(stage_path or STAGE23,columns=['candidate_id','source_row_number','source_shard_path','path_end_exclusive'])
 pieces=[]
 for p in fs:
  raw=pd.read_parquet(p,columns=['__ts__','__symbol__','side_name',*RAW_FEATURES]).copy();raw['source_row_number']=np.arange(len(raw),dtype=np.int64)
  key=str(p.resolve());ids=stage.loc[stage.source_shard_path.eq(key),['candidate_id','source_row_number','path_end_exclusive']]
  # The raw backcast shard contains unselected diagnostic rows too; retain only
  # the rows sealed in the exact candidate-stage ledger.
  raw=raw.merge(ids,on='source_row_number',how='inner',validate='one_to_one').drop(columns='source_row_number').rename(columns={'path_end_exclusive':'execution_label_end_utc'})
  pieces.append(raw)
 return pd.concat(pieces,ignore_index=True)

def panel()->pd.DataFrame:
 raw=pd.concat([_read_raw(RAW23,historical=True,stage_path=STAGE23),_read_raw(RAW24,historical=True,stage_path=STAGE24)],ignore_index=True)
 raw=validate_candidate_identity(raw)
 oof=validate_candidate_identity(pd.read_parquet(OOF))
 oof=oof.loc[oof['__ts__'].dt.year.le(2024)].copy()
 x=raw.merge(oof,on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one')
 tr=pd.read_parquet(TRANS,columns=[*IDENTITY_COLUMNS,'transition_active_probability','transition_available_utc','transition_train_end_utc'])
 tr=validate_candidate_identity(tr).rename(columns={'transition_active_probability':'p_transition_active_oof'})
 x=x.merge(tr,on=list(IDENTITY_COLUMNS),how='left',validate='one_to_one')
 x=validate_candidate_identity(x).sort_values(['__ts__','candidate_id'],kind='stable').reset_index(drop=True)
 for c in ['execution_label_end_utc','transition_available_utc','transition_train_end_utc']:
  if c in x: x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
 bad=x.loc[x['__ts__'].dt.year.eq(2024)&x['p_transition_active_oof'].notna()&x['transition_available_utc'].gt(x['__ts__'])]
 if len(bad): raise RegimeOOFStackError('transition output is not as-of candidate timestamp')
 return x

def mat(x:pd.DataFrame, cols:list[str], med:pd.Series|None=None)->tuple[np.ndarray,pd.Series]:
 a=x[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
 med=a.median().fillna(0.) if med is None else med
 return a.fillna(med).to_numpy(np.float32),med

class DAE(nn.Module):
 def __init__(self,n:int):
  super().__init__();self.enc=nn.Sequential(nn.Linear(n,24),nn.ReLU(),nn.Linear(24,8));self.dec=nn.Sequential(nn.ReLU(),nn.Linear(8,24),nn.ReLU(),nn.Linear(24,n))
 def forward(self,x): return self.dec(self.enc(x))

def representations(train:pd.DataFrame, ev:pd.DataFrame, seed:int)->tuple[pd.DataFrame,pd.DataFrame,dict]:
 x,med=mat(train,RAW_FEATURES); z,_=mat(ev,RAW_FEATURES,med)
 scaler=StandardScaler().fit(x); x=scaler.transform(x).astype('float32');z=scaler.transform(z).astype('float32')
 rng=np.random.default_rng(seed);take=rng.choice(len(x),size=min(len(x),30000),replace=False)
 fit=x[take]
 gmm=GaussianMixture(n_components=8,covariance_type='diag',max_iter=60,n_init=1,random_state=seed).fit(fit)
 def geo(a:np.ndarray)->np.ndarray:
  post=gmm.predict_proba(a);d=((a[:,None,:]-gmm.means_[None,:,:])**2/gmm.covariances_[None,:,:]).sum(2)
  return np.c_[-gmm.score_samples(a),d.min(1),np.sum(post*d,axis=1)]
 gx,gz=geo(x),geo(z)
 torch.manual_seed(seed);net=DAE(x.shape[1]);opt=torch.optim.Adam(net.parameters(),lr=1e-3);loss=nn.MSELoss();t=torch.from_numpy(fit)
 net.train()
 for _ in range(4):
  for i in torch.randperm(len(t)).split(512):
   clean=t[i];noisy=clean+0.03*torch.randn_like(clean);opt.zero_grad();v=loss(net(noisy),clean);v.backward();opt.step()
 net.eval()
 with torch.no_grad():
  tx,tz=torch.from_numpy(x),torch.from_numpy(z);hx=net.enc(tx).numpy();hz=net.enc(tz).numpy();ex=((net(tx)-tx)**2).mean(1).numpy();ez=((net(tz)-tz)**2).mean(1).numpy()
 mu,sd=float(ex.mean()),max(float(ex.std()),1e-6)
 def frame(g,h,e,index):
  q=pd.DataFrame(g,index=index,columns=['gmm_ood_score','mahalanobis_distance','expected_mahalanobis'])
  for i in range(8):q[f'dae_fold_{i:02d}']=h[:,i]
  q['dae_reconstruction_error_zscore']=(e-mu)/sd
  return q
 return frame(gx,hx,ex,train.index),frame(gz,hz,ez,ev.index),{'gmm_fit_rows':len(fit),'dae_fit_rows':len(fit),'dae_epochs':4,'raw_feature_count':len(RAW_FEATURES)}

def failure_probability(train:pd.DataFrame,ev:pd.DataFrame,seed:int)->tuple[np.ndarray,np.ndarray]:
 x,med=mat(train,RAW_FEATURES);z,_=mat(ev,RAW_FEATURES,med);y=train[TARGET].lt(0).astype(int).to_numpy();order=np.argsort(train['__ts__'].to_numpy());o=np.full(len(train),float(y.mean()),dtype='float32')
 # Chronological inner OOF supplies the outer execution model without own-row outcome fitting.
 cuts=np.array_split(order,4)
 for j in range(1,len(cuts)):
  fit=np.concatenate(cuts[:j]);va=cuts[j]
  if len(fit)<1000: continue
  m=lgb.LGBMClassifier(n_estimators=100,learning_rate=.05,num_leaves=15,min_child_samples=150,reg_lambda=3.,random_state=seed+j,n_jobs=4,verbosity=-1).fit(x[fit],y[fit]);o[va]=m.predict_proba(x[va])[:,1]
 m=lgb.LGBMClassifier(n_estimators=120,learning_rate=.05,num_leaves=15,min_child_samples=150,reg_lambda=3.,random_state=seed,n_jobs=4,verbosity=-1).fit(x,y)
 return o,np.asarray(m.predict_proba(z)[:,1],dtype='float32')

def mapper(x:np.ndarray,y:np.ndarray):
 ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok]
 if len(x)<8 or np.unique(x).size<2:
  v=float(y.mean()) if len(y) else 0.;return lambda a:np.full(len(a),v)
 m=IsotonicRegression(out_of_bounds='clip',increasing='auto').fit(x,y);return lambda a:np.asarray(m.predict(np.asarray(a,float)),float)

def rank(a:pd.Series,b:pd.Series)->float:
 a,b=pd.to_numeric(a,errors='coerce'),pd.to_numeric(b,errors='coerce');ok=a.notna()&b.notna();return float(a[ok].rank().corr(b[ok].rank())) if ok.sum()>2 else float('nan')

def run(output:Path=OUT)->Path:
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 x=panel();pred={a:[] for a in ARMS};diagnostics=[];audit=[]
 for n,start in enumerate(pd.date_range('2024-01-01','2024-12-01',freq='MS',tz='UTC')):
  end=start+pd.offsets.MonthBegin(1);ev=x.loc[(x.__ts__>=start)&(x.__ts__<end)].copy();tr=x.loc[x.execution_label_end_utc.lt(start)].copy()
  if len(ev)==0 or len(tr)<12000: raise RegimeOOFStackError(f'insufficient exact common support for {start}')
  a,b,rep_audit=representations(tr,ev,313+n);tr=tr.join(a);ev=ev.join(b)
  pf_train,pf_ev=failure_probability(tr,ev,811+n);tr['p_failure_destination_12h']=pf_train;ev['p_failure_destination_12h']=pf_ev
  tr['p_transition_active_oof']=tr['p_transition_active_oof'].fillna(0.);ev['p_transition_active_oof']=ev['p_transition_active_oof'].fillna(0.)
  rec={'fold_id':f'month_{start:%Y%m}','evaluation_start_utc':start,'evaluation_end_exclusive_utc':end,'train_rows':len(tr),'evaluation_rows':len(ev),'train_label_end_max':tr.execution_label_end_utc.max(),**rep_audit}
  diagnostics.append(ev.loc[:,[*IDENTITY_COLUMNS,*RETAINED_DIAGNOSTICS]].assign(representation_fold_id=rec['fold_id'],representation_train_rows=len(tr),representation_train_label_end_max=tr.execution_label_end_utc.max(),representation_is_fold_local=True))
  for ai,(arm,cols) in enumerate(ARMS.items()):
   out=[]
   for side,local in ev.groupby('side_name',observed=True):
    fit=tr.loc[tr.side_name.eq(side)];xx,med=mat(fit,cols);zz,_=mat(local,cols,med);y=fit[TARGET].to_numpy(float)
    m=lgb.LGBMRegressor(n_estimators=180,learning_rate=.035,num_leaves=15,min_child_samples=180,subsample=.85,colsample_bytree=.9,reg_lambda=3.,random_state=1000+n*17+ai,n_jobs=4,verbosity=-1).fit(xx,y)
    recent=fit.loc[fit.execution_label_end_utc.ge(fit.execution_label_end_utc.max()-pd.Timedelta(days=21))];xr,_=mat(recent,cols,med);mp=mapper(m.predict(xr),recent[TARGET].to_numpy(float));raw=m.predict(zz)
    out.append(local.loc[:,list(IDENTITY_COLUMNS)].assign(extension_fold_id=rec['fold_id'],raw_score=raw,mapped_score=mp(raw)))
   pred[arm].append(pd.concat(out,ignore_index=True))
  audit.append(rec)
 tmp=Path(tempfile.mkdtemp(dir=output.parent,prefix=f'.{output.name}.'))
 try:
  sidecar=tmp/'prediction_sidecars';sidecar.mkdir();frames={}
  for arm,parts in pred.items():
   q=pd.concat(parts,ignore_index=True);validate_candidate_identity(q);q.to_parquet(sidecar/f'{arm}.parquet',index=False);frames[arm]=x.merge(q,on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one')
  cols=EvaluationColumns(mapped_score='mapped_score',alpha_target=ALPHA,net_ev=TARGET,gross_ev='execution_gross_ev_12h',cost='execution_cost_return')
  summary,periods,_=evaluate_matched_arms(frames,columns=cols,top_fraction=.10,category_col=None)
  base=summary.loc[summary.arm.eq('baseline')].iloc[0]
  for arm in ARMS:
   row=summary.arm.eq(arm);months=periods.loc[(periods.arm==arm)&(periods.period_type=='month')].sort_values('period');latest=months.tail(1).iloc[0]
   summary.loc[row,'latest_month']=latest.period;summary.loc[row,'latest_month_net_ev']=latest.mean_net_ev;summary.loc[row,'aggregate_incremental_net_ev_vs_baseline']=summary.loc[row,'top10_mean_net_ev'].iloc[0]-base.top10_mean_net_ev
   b_latest=periods.loc[(periods.arm=='baseline')&(periods.period_type=='month')&(periods.period==latest.period),'mean_net_ev'].iloc[0];summary.loc[row,'latest_incremental_net_ev_vs_baseline']=latest.mean_net_ev-b_latest
   summary.loc[row,'aggregate_and_latest_gate_pass']=bool(summary.loc[row,'top10_mean_net_ev'].iloc[0]>0 and latest.mean_net_ev>0 and summary.loc[row,'aggregate_incremental_net_ev_vs_baseline'].iloc[0]>0 and summary.loc[row,'latest_incremental_net_ev_vs_baseline'].iloc[0]>0)
  pd.DataFrame(audit).to_parquet(tmp/'fold_provenance.parquet',index=False);summary.to_csv(tmp/'metrics_summary.csv',index=False);periods.to_parquet(tmp/'period_metrics.parquet',index=False);diag=pd.concat(diagnostics,ignore_index=True);validate_candidate_identity(diag);diag.to_parquet(tmp/'fold_local_representation_sidecar.parquet',index=False);(tmp/'feature_contract.json').write_text(json.dumps({'raw_causal_features':RAW_FEATURES,'arms':ARMS,'retained_per_candidate_diagnostics':RETAINED_DIAGNOSTICS,'excluded':'all raw precomputed GMM posterior/entropy/compact-risk fields and all action/timing/MAE/target-price/wait fields','gmm':'fit only on pre-fold candidates; geometry only','dae':'fold-local denoising autoencoder trained only pre-fold','failure_destination':'inner-chronological-OOF p(execution_net_ev_12h < 0); outer eval model fit only pre-fold','transition':'sealed pre-block reconstructed transition_active_probability; unavailable pre-2024 rows deterministically zero-filled'},indent=2,sort_keys=True)+'\n')
  files=[tmp/'fold_provenance.parquet',tmp/'metrics_summary.csv',tmp/'period_metrics.parquet',tmp/'fold_local_representation_sidecar.parquet',tmp/'feature_contract.json',*sorted(sidecar.glob('*.parquet'))]
  man={'schema':SCHEMA,'status':'DIAGNOSTIC_HISTORICAL_BACKCAST_COMPLETE','coverage':'full calendar 2024; historical raw training begins 2022-08; exact current-spread-counterfactual 1m economics','limitations':'not deployed execution parity; no all-era claim; promotion and portfolio replay forbidden','selection':'one pooled global monthly top10 after each arm own causal trailing-21d EV map','common_2024_candidate_rows':int(sum(len(f) for f in frames.values())/len(frames)),'folds':len(audit),'promotion_eligible':False,'portfolio_replay':False,'inputs':{str(p.resolve()):sha(p) for p in [OOF,TRANS,*sorted(RAW23.glob('candidates_*.parquet')),*sorted(RAW24.glob('candidates_*.parquet'))]},'outputs_sha256':{str(p.relative_to(tmp)):sha(p) for p in files}}
  mp=tmp/'manifest.json';mp.write_text(json.dumps(man,indent=2,sort_keys=True,default=str)+'\n');(tmp/'manifest.sha256').write_text(f'{sha(mp)}  manifest.json\n');os.replace(tmp,output);return output
 except Exception: shutil.rmtree(tmp,ignore_errors=True);raise

def parse(argv:Sequence[str]|None=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);return p.parse_args(argv)
if __name__=='__main__': print(run(**vars(parse())))
