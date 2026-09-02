#!/usr/bin/env python3
"""Frozen-score attribution of old/new global-top10 exact-EV failures.

Transition event labels and path outcomes are diagnostic-only joins performed
after score freeze.  The final feature table contains only fields observed at
or before the signal timestamp.
"""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SCORES=ROOT/'data_perp/artifacts/febapr2025_native12h_execution_ev_divergence_20260729_v1/joined_scores_execution_ev.parquet'
POP=ROOT/'data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet'
LABELS=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
OUT=ROOT/'data_perp/artifacts/febapr2025_native12_execution_ev_failure_attribution_20260729_v4'
FEATURES=('__regime_vol_12h__','__regime_trend_12h__','__meta_raw__chop_score','__meta_raw__volatility_zscore','jump_intensity','breakout_24h','range_24h_pct','spread_proxy_abs_return_bps_robust_z','trend_r2_24','path_efficiency_24')
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def phase(x:pd.DataFrame)->pd.Series:
 return np.where(x.expost_transition_active.eq(1),'active',np.where(x.transition_window_member.eq(1),'window_nonactive','outside'))
def select(x:pd.DataFrame,col:str)->pd.Series:
 k=max(1,int(np.ceil(len(x)*.1)));out=pd.Series(False,index=x.index);out.loc[x.nlargest(k,col).index]=True;return out
def paired(x:pd.DataFrame)->pd.Series:
 return pd.Series(np.where(x.old_selected & x.new_selected,'both',np.where(x.old_selected,'old_only',np.where(x.new_selected,'new_only','neither'))),index=x.index)
def grouped(x:pd.DataFrame,field:str)->pd.DataFrame:
 rows=[]
 for value,g in x.groupby(field,dropna=False,sort=True):
  row={field:str(value),'support':len(g)}
  for arm in ('old','new'):
   s=g[f'{arm}_selected'];top=g[s];fail=top.execution_net_ev_12h.le(0)
   row.update({f'{arm}_selected':int(s.sum()),f'{arm}_failure':int(fail.sum()),f'{arm}_failure_rate':float(fail.mean()) if len(top) else np.nan,f'{arm}_net_mean':float(top.execution_net_ev_12h.mean()) if len(top) else np.nan})
  row['selected_delta_new_minus_old']=row['new_selected']-row['old_selected'];row['failure_delta_new_minus_old']=row['new_failure']-row['old_failure'];row['net_delta_new_minus_old']=row['new_net_mean']-row['old_net_mean']
  rows.append(row)
 return pd.DataFrame(rows)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 score=pd.read_parquet(SCORES);pcols=['candidate_id','transition_event_id','expost_transition_active','transition_window_member','execution_exit_reason','execution_exit_minute','execution_mfe_return_12h','execution_mae_return_12h','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h']
 pop=pd.read_parquet(POP,columns=pcols);x=score.merge(pop,on='candidate_id',how='inner',validate='one_to_one',suffixes=('','__pop'))
 if len(x)!=509868 or x.candidate_id.duplicated().any():raise ValueError('frozen score identity contract fails')
 for col in ('execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h'):
  if not np.allclose(x[col],x[f'{col}__pop'],atol=1e-12,rtol=0,equal_nan=True):raise ValueError(f'frozen score/population {col} mismatch')
 if not np.allclose(x.execution_gross_ev_12h-x.execution_cost_return,x.execution_net_ev_12h,atol=1e-12,rtol=0,equal_nan=True):raise ValueError('gross - cost != net')
 x['transition_phase']=phase(x);x['old_selected']=select(x,'old_score');x['new_selected']=select(x,'new_score');x['selection_transition']=paired(x)
 x['cost_bucket']=pd.qcut(x.execution_cost_return,4,duplicates='drop').astype(str);x['mfe_bucket']=pd.qcut(x.execution_mfe_return_12h,4,duplicates='drop').astype(str);x['mae_bucket']=pd.qcut(x.execution_mae_return_12h,4,duplicates='drop').astype(str);x['exit_time_bucket']=pd.cut(x.execution_exit_minute,[-1,60,180,360,720],labels=('<=60m','61-180m','181-360m','361-720m')).astype(str)
 tables={name:grouped(x,name) for name in ('transition_phase','transition_event_id','side_name','month','__symbol__','cost_bucket','execution_exit_reason','mfe_bucket','mae_bucket','exit_time_bucket')}
 transition=x.groupby('selection_transition',sort=True).agg(support=('candidate_id','size'),gross_mean=('execution_gross_ev_12h','mean'),cost_mean=('execution_cost_return','mean'),net_mean=('execution_net_ev_12h','mean'),positive_net_rate=('execution_net_ev_12h',lambda z:float((z>0).mean()))).reset_index()
 only=transition.set_index('selection_transition')
 displacement={f'new_only_minus_old_only_{col}':float(only.loc['new_only',col]-only.loc['old_only',col]) for col in ('gross_mean','cost_mean','net_mean','positive_net_rate')}
 # Causal, signal-time feature contrasts; ex-post outcomes remain diagnostic only.
 cols=['candidate_id',*FEATURES];raw=[];label_sources={}
 for m in (2,3,4):
  for s in ('long','short'):
   path=LABELS/f'train_global_{s}_5_2025_{m:02d}.parquet';available=[c for c in cols if c in pd.read_parquet(path,columns=None).columns];label_sources[str(path)]=sha(path);raw.append(pd.read_parquet(path,columns=available))
 f=x.loc[:,['candidate_id','execution_net_ev_12h','selection_transition','new_selected']].merge(pd.concat(raw,ignore_index=True),on='candidate_id',how='left',validate='one_to_one');f['failure']=f.execution_net_ev_12h.le(0)
 feature_rows=[]
 for comparison,left,right in (('new_selected_failure_vs_success',f.new_selected & f.failure,f.new_selected & ~f.failure),('new_only_vs_old_only',f.selection_transition.eq('new_only'),f.selection_transition.eq('old_only'))):
  for col in FEATURES:
   if col not in f:continue
   a=pd.to_numeric(f.loc[left,col],errors='coerce').dropna();b=pd.to_numeric(f.loc[right,col],errors='coerce').dropna();sd=float(f.loc[left|right,col].std())
   feature_rows.append({'comparison':comparison,'feature':col,'left_support':len(a),'right_support':len(b),'left_mean':float(a.mean()),'right_mean':float(b.mean()),'standardized_difference':float((a.mean()-b.mean())/sd) if sd>0 else np.nan})
 features=pd.DataFrame(feature_rows);features['abs_standardized_difference']=features.standardized_difference.abs();features=features.sort_values(['comparison','abs_standardized_difference'],ascending=[True,False])
 temp=Path(tempfile.mkdtemp(dir=OUT.parent,prefix=f'.{OUT.name}.'))
 for name,table in tables.items():table.to_parquet(temp/f'{name}.parquet',index=False,compression='zstd')
 transition.to_parquet(temp/'paired_selection_transition.parquet',index=False,compression='zstd');features.to_parquet(temp/'causal_preentry_feature_contrasts.parquet',index=False,compression='zstd');x.to_parquet(temp/'joined_frozen_attribution_rows.parquet',index=False,compression='zstd')
 identity=hashlib.sha256(pd.util.hash_pandas_object(x.candidate_id.astype(str).sort_values(kind='stable'),index=False).values.tobytes()).hexdigest()
 manifest={'schema':'native12_execution_ev_failure_attribution_v4','status':'DIAGNOSTIC_ONLY_FROZEN_SCORES','rows':len(x),'selection':'one pooled global top10 per score; no post-selected policy','causality':{'scores':'frozen before all joins','transition_labels':'ex-post diagnostic only','path_outcomes':'ex-post diagnostic only','feature_contrasts':'pre-entry archived signal-time fields only'},'identity':{'candidate_id_sha256':identity,'matches_frozen_divergence_identity':identity=='fe6dfe0fd4054fa83b25178af1ccc8e45b2d247a0c92264cf64eccd51bb41daa','unique_rows':int(x.candidate_id.nunique())},'accounting_assertions':{'frozen_score_and_population_cost_fields_agree_atol':1e-12,'gross_minus_cost_equals_net_atol':1e-12,'paired_displacement_new_only_minus_old_only':displacement},'sources':{str(SCORES):sha(SCORES),str(POP):sha(POP),**label_sources},'outputs_sha256':{p.name:sha(p) for p in sorted(temp.glob('*.parquet'))},'checksum_convention':'Each material output is SHA256-listed here; manifest.json is checked by the detached manifest.sha256 sidecar (the sidecar cannot hash itself).'}
 (temp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(temp/'manifest.sha256').write_text(f'{sha(temp / "manifest.json")}  manifest.json\n');os.replace(temp,OUT)
if __name__=='__main__':main()
