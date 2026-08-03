#!/usr/bin/env python3
"""Seal a bounded, schema-aware diagnosis of DAE non-transfer (2024 vs 2026).

It never treats precomputed historical embeddings as the fold-local DAE used in
the 2024 economic test.  They are diagnostic proxies only; the 2024 artifact
did not retain per-row fold-local embeddings.  Therefore this report can test
score/economics non-transfer and representation distribution shift, but cannot
validate a causal representation-trust gate.
"""
from __future__ import annotations
import hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity # noqa:E402

OUT=ROOT/'data_perp/artifacts/dae_cross_era_nontransfer_diagnosis_20260730_v2'
X24=ROOT/'data_perp/artifacts/unsupervised_economic_2024_extension_20260730_v1'
X26=ROOT/'data_perp/artifacts/unsupervised_economic_common_oof_20260730_v2'
RAW24=ROOT/'data_perp/reports/failure_2024_transition_exact1m_candidate_backcast_20260730_v1/candidate_shards'
STAGE24=ROOT/'data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2/staged_candidates.parquet'
REP26=ROOT/'data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet'
OOF24=ROOT/'data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet'
LAT=[f'dae_b16_{i:02d}' for i in range(16)]
REP=[*LAT,'dae_reconstruction_error_zscore','gmm_ood_score','mahalanobis_distance']

def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def load24proxy():
 stage=pd.read_parquet(STAGE24,columns=['candidate_id','source_row_number','source_shard_path'])
 ps=[]
 for p in sorted(RAW24.glob('candidates_*.parquet')):
  raw=pd.read_parquet(p,columns=['__ts__','__symbol__','side_name',*REP]);raw['source_row_number']=np.arange(len(raw))
  ids=stage.loc[stage.source_shard_path.eq(str(p.resolve())),['candidate_id','source_row_number']]
  ps.append(raw.merge(ids,on='source_row_number',how='inner',validate='one_to_one').drop(columns='source_row_number'))
 return validate_candidate_identity(pd.concat(ps,ignore_index=True))

def scores(root:Path,era:str,base:str='baseline',dae:str='dae_only'):
 b=pd.read_parquet(root/'prediction_sidecars'/f'{base}.parquet');d=pd.read_parquet(root/'prediction_sidecars'/f'{dae}.parquet')
 q=b.merge(d,on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one',suffixes=('_base','_dae'))
 q['era']=era;q['month']=q['__ts__'].dt.strftime('%Y-%m');q['score_delta']=q.mapped_score_dae-q.mapped_score_base
 return q

def rank(a,b):
 a=pd.to_numeric(a,errors='coerce');b=pd.to_numeric(b,errors='coerce');ok=a.notna()&b.notna();return float(a[ok].rank().corr(b[ok].rank())) if ok.sum()>3 else np.nan

def run(output:Path=OUT):
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 p24=load24proxy();p26=pd.read_parquet(REP26,columns=[*IDENTITY_COLUMNS,*REP,'execution_net_ev_12h'])
 # This is a schema fact: there is no shared raw causal market field between surfaces.
 raw_cols=set(pd.read_parquet(sorted(RAW24.glob('*.parquet'))[0]).columns);rep_cols=set(pd.read_parquet(REP26).columns)
 shared=sorted(raw_cols&rep_cols);shared_non_rep=[c for c in shared if c not in set(IDENTITY_COLUMNS) and not any(t in c.lower() for t in ('dae','gmm','cluster','mahal','entropy'))]
 s24=scores(X24,'2024');s26=scores(X26,'2026')
 o24=pd.read_parquet(OOF24,columns=[*IDENTITY_COLUMNS,'execution_net_ev_12h','__reconstructed_soft_alpha_12h__'])
 s24=s24.merge(o24,on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one')
 s26=s26.merge(p26.loc[:,[*IDENTITY_COLUMNS,'execution_net_ev_12h']],on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one')
 r24=p24.merge(s24.loc[:,[*IDENTITY_COLUMNS,'era','month','score_delta','mapped_score_base','mapped_score_dae','execution_net_ev_12h']],on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one')
 r26=p26.merge(s26.loc[:,[*IDENTITY_COLUMNS,'era','month','score_delta','mapped_score_base','mapped_score_dae']],on=list(IDENTITY_COLUMNS),how='inner',validate='one_to_one')
 allx=pd.concat([r24,r26],ignore_index=True);allx['side_name']=allx.side_name.astype(str)
 dist=[];assoc=[];cov=[]
 for (era,month,side),q in allx.groupby(['era','month','side_name'],observed=True):
  for c in REP:
   v=pd.to_numeric(q[c],errors='coerce');dist.append({'era':era,'month':month,'side_name':side,'field':c,'rows':len(q),'mean':v.mean(),'std':v.std(ddof=0),'q10':v.quantile(.1),'q50':v.quantile(.5),'q90':v.quantile(.9)})
  assoc.append({'era':era,'month':month,'side_name':side,'rows':len(q),'score_delta_net_rank_ic':rank(q.score_delta,q.execution_net_ev_12h),'baseline_net_rank_ic':rank(q.mapped_score_base,q.execution_net_ev_12h),'dae_net_rank_ic':rank(q.mapped_score_dae,q.execution_net_ev_12h)})
  a=q[LAT].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(float);c=np.corrcoef(a,rowvar=False);cov.append({'era':era,'month':month,'side_name':side,'rows':len(q),'latent_abs_corr_mean':float(np.mean(np.abs(c[np.triu_indices_from(c,1)]))),'latent_cov_trace':float(np.trace(np.cov(a,rowvar=False)))})
 pm=[]
 for era,root in [('2024',X24),('2026',X26)]:
  z=pd.read_parquet(root/'period_metrics.parquet');b=z.loc[(z.arm=='baseline')&(z.period_type=='month'),['period','mean_net_ev','execution_net_rank_ic']].rename(columns={'mean_net_ev':'baseline_net_ev','execution_net_rank_ic':'baseline_execution_ic'})
  d=z.loc[(z.arm=='dae_only')&(z.period_type=='month'),['period','mean_net_ev','execution_net_rank_ic']].rename(columns={'mean_net_ev':'dae_net_ev','execution_net_rank_ic':'dae_execution_ic'})
  t=b.merge(d,on='period');t['era']=era;t['dae_incremental_net_ev']=t.dae_net_ev-t.baseline_net_ev;t['dae_incremental_execution_ic']=t.dae_execution_ic-t.baseline_execution_ic;pm.append(t)
 pm=pd.concat(pm,ignore_index=True);a=pd.DataFrame(assoc);tr=[]
 for era,q in a.groupby('era'):
  tr.append({'era':era,'months':q.month.nunique(),'median_delta_net_rank_ic':q.score_delta_net_rank_ic.median(),'positive_month_fraction_score_delta_ic':q.score_delta_net_rank_ic.gt(0).mean()})
 tmp=Path(tempfile.mkdtemp(dir=output.parent,prefix=f'.{output.name}.'))
 try:
  pd.DataFrame(dist).to_parquet(tmp/'representation_distribution_proxy.parquet',index=False);pd.DataFrame(cov).to_parquet(tmp/'latent_covariance_proxy.parquet',index=False);a.to_parquet(tmp/'score_association_by_month_side.parquet',index=False);pm.to_csv(tmp/'dae_economic_uplift_by_month.csv',index=False);pd.DataFrame(tr).to_csv(tmp/'association_summary.csv',index=False)
  prereg={'status':'PRE_REGISTERED_NEXT_ABLATIONS_ONLY','rule':'No direct promotion/gate from this diagnostic. Require full-2024 and full-2026 matched fold-local embeddings saved per candidate, then fit trust only on prior months and test locked later months.','ablations':['save per-candidate fold-local DAE code and reconstruction error plus fold identifier','add causal representation-age/train-support and OOD features; test trust gate versus no gate','test DAE only inside high-support/low-OOD strata without GMM posterior or action inputs','test era-balanced train-only DAE versus rolling-window DAE; report aggregate/latest-month economics and IC','pre-register threshold selection on prior months only and evaluate a frozen later-month global top10'], 'minimum_evidence':'two disjoint later periods with positive incremental net EV and stable monthly Q10; otherwise reject trust gate'}
  (tmp/'pre_registered_next_ablations.json').write_text(json.dumps(prereg,indent=2)+'\n')
  blocker={'shared_columns_raw2024_vs_rep2026':shared,'shared_non_representation_causal_raw_fields':shared_non_rep,'conclusion':'No shared causal raw market fields are present in the two sealed surfaces. 2024 fold-local DAE embeddings were not retained; only historical precomputed embedding proxies can be distribution-compared. A causal representation-trust signal is therefore NOT validated.'}
  (tmp/'schema_and_identifiability_blocker.json').write_text(json.dumps(blocker,indent=2)+'\n')
  files=sorted(p for p in tmp.iterdir() if p.is_file());man={'schema':'dae_cross_era_nontransfer_diagnosis_v1','status':'BOUNDED_DIAGNOSTIC_COMPLETE_NO_TRUST_GATE','coverage':'2024 full calendar diagnostic extension vs May-July 2026 common OOF slice','limitations':['2024 fold-local DAE per-row embeddings were not retained','2024 precomputed embedding fields are proxy diagnostics only and cannot identify the fold-local DAE coordinates','no shared causal raw market feature fields across surfaces','2026 is partial May-July, so no all-era inference'],'promotion_eligible':False,'inputs':{str(p.resolve()):sha(p) for p in [X24/'manifest.json',X26/'manifest.json',REP26,OOF24,STAGE24]},'outputs_sha256':{p.name:sha(p) for p in files}}
  mp=tmp/'manifest.json';mp.write_text(json.dumps(man,indent=2,sort_keys=True)+'\n');(tmp/'manifest.sha256').write_text(f'{sha(mp)}  manifest.json\n');os.replace(tmp,output);return output
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':print(run())
