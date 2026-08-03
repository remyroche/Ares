#!/usr/bin/env python3
"""Final supervised topology-taxonomy admissibility audit.

Before fitting any classifier, prove that state 0/nonzero and state identities
mean the same thing across 2022--25.  Candidate labels are ex-post targets
only: baseline->nonbaseline onset, nonbaseline->baseline normalization, and
nonbaseline->different-nonbaseline rotation.  Destination/type never enters a
causal predictor.  If semantic alignment or label support fails, no subtype
model is trained and the sealed result terminates this bounded workstream.
"""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1'
CAT=ROOT/'data_perp/artifacts/transition_pattern_catalogue_20260730_v6'
OUT=ROOT/'data_perp/artifacts/supervised_coarse_topology_taxonomy_audit_20260730_v1'
SPLIT=pd.Timestamp('2026-01-01',tz='UTC')
FEATURES=['breadth_dispersion','broad_washout_recovery','btc_resilience_alt_weakness','correlation_breakdown_dispersion','deleveraged_range_climax_reversal','deleveraging_without_followthrough','downside_breadth_intensity','funding_confirmed_long_flush','funding_confirmed_short_covering','funding_deleveraging_divergence','peer_volatility_decoupling','short_breakout_exhaustion']
MIN_STATE_HOURS=50;MIN_CLASS_EVENTS=12;MIN_CLASS_ERAS=3;MIN_CONTRAST_CORR=.50;MIN_STATE_CORR=.40
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p:Path,x:object):
 q=p.with_name('.'+p.name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def corrs(profiles:dict[int,pd.Series],kind:str,state:object)->list[dict]:
 rows=[]
 for left,right in combinations(sorted(profiles),2):rows.append({'kind':kind,'state':state,'era_a':left,'era_b':right,'spearman_profile_correlation':float(profiles[left].corr(profiles[right],method='spearman'))})
 return rows
def taxonomy_label(source:pd.Series,destination:pd.Series)->pd.Series:
 return pd.Series(np.select([(source.eq(0)&destination.ne(0)),(source.ne(0)&destination.eq(0)),(source.ne(0)&destination.ne(0)&source.ne(destination))],['baseline_to_nonbaseline_onset','nonbaseline_to_baseline_normalization','nonbaseline_to_nonbaseline_rotation'],'other_abstain'),index=source.index)
def run(output:Path=OUT)->Path:
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 hourly=pd.read_parquet(LEDGER/'hourly_state_calendar.parquet');hourly['source_utc']=pd.to_datetime(hourly.source_utc,utc=True)
 if hourly.source_utc.duplicated().any() or (hourly.source_utc.astype('int64')%pd.Timedelta(hours=1).value!=0).any():raise ValueError('nonhourly panel')
 train=hourly[hourly.source_utc<SPLIT].copy();train['era']=train.source_utc.dt.year.astype(int);state=pd.to_numeric(train.target__pooled_state,errors='raise').astype(int)
 med=train[FEATURES].median();mad=(train[FEATURES]-med).abs().median()*1.4826;mad=mad.mask(mad<1e-6,1.);z=(train[FEATURES]-med)/mad
 z['era']=train.era;z['state']=state
 supports=z.groupby(['era','state'],observed=True).size().rename('hours').reset_index();state_profiles=[];identity={};contrast={}
 for era,g in z.groupby('era',observed=True):
  zero=g[g.state.eq(0)][FEATURES].median();nonzero=g[g.state.ne(0)][FEATURES].median();contrast[int(era)]=zero-nonzero
  for s,h in g.groupby('state',observed=True):
   if len(h)>=MIN_STATE_HOURS:identity.setdefault(int(s),{})[int(era)]=h[FEATURES].median()
   state_profiles.extend({'era':int(era),'state':int(s),'feature':f,'hours':len(h),'robust_profile':float(h[f].median())} for f in FEATURES)
 contrast_rows=corrs(contrast,'baseline_zero_minus_nonzero','0_vs_nonzero');id_rows=[]
 for s,p in identity.items():id_rows.extend(corrs(p,'state_id_profile',s))
 correlations=pd.DataFrame([*contrast_rows,*id_rows])
 modal=supports.sort_values(['era','hours'],ascending=[True,False]).groupby('era',as_index=False).first();modal['is_state_zero']=modal.state.eq(0);zero_share=supports[supports.state.eq(0)].merge(supports.groupby('era',as_index=False).hours.sum().rename(columns={'hours':'total_hours'}),on='era');zero_share['zero_share']=zero_share.hours/zero_share.total_hours
 contrast_min=float(pd.DataFrame(contrast_rows).spearman_profile_correlation.min());id_summary=[]
 for s,p in identity.items():
  r=pd.DataFrame(corrs(p,'state_id_profile',s));id_summary.append({'state':s,'supported_eras':len(p),'minimum_pairwise_profile_correlation':float(r.spearman_profile_correlation.min()) if len(r) else np.nan,'passes_identity':bool(len(p)>=MIN_CLASS_ERAS and len(r) and r.spearman_profile_correlation.min()>=MIN_STATE_CORR)})
 id_summary=pd.DataFrame(id_summary);semantic_pass=bool(modal.is_state_zero.all() and (zero_share.zero_share>.5).all() and contrast_min>=MIN_CONTRAST_CORR and id_summary.passes_identity.all())
 events=pd.read_parquet(LEDGER/'transition_episode_ledger.parquet');events['anchor_source_utc']=pd.to_datetime(events.anchor_source_utc,utc=True);events['era']=events.anchor_source_utc.dt.year.astype(int);events['topology_target']=taxonomy_label(pd.to_numeric(events.source_state),pd.to_numeric(events.destination_state))
 seq=pd.read_parquet(CAT/'event_preonset_sequences.parquet',columns=['event_id','anchor_source_utc']);seq['anchor_source_utc']=pd.to_datetime(seq.anchor_source_utc,utc=True)
 events=events.merge(seq,on=['event_id','anchor_source_utc'],how='inner',validate='one_to_one');anchor_state=events.merge(hourly[['source_utc','target__pooled_state']],left_on='anchor_source_utc',right_on='source_utc',how='left',validate='one_to_one');state_match=float(pd.to_numeric(anchor_state.source_state).eq(pd.to_numeric(anchor_state.target__pooled_state)).mean())
 top_support=events.groupby(['era','topology_target'],observed=True).size().rename('events').reset_index();train_top=top_support[top_support.era.lt(2026)];gate=train_top.groupby('topology_target',observed=True).agg(events=('events','sum'),eras=('era','nunique')).reset_index();gate['passes_support']=gate.events.ge(MIN_CLASS_EVENTS)&gate.eras.ge(MIN_CLASS_ERAS)
 expected={'baseline_to_nonbaseline_onset','nonbaseline_to_baseline_normalization','nonbaseline_to_nonbaseline_rotation','other_abstain'};missing=expected-set(gate.topology_target);label_pass=not missing and gate.set_index('topology_target').loc[list(expected),'passes_support'].all()
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  supports.to_csv(stage/'state_hour_support_by_era.csv',index=False);pd.DataFrame(state_profiles).to_csv(stage/'state_semantic_profiles.csv',index=False);correlations.to_csv(stage/'state_profile_stability_correlations.csv',index=False);modal.to_csv(stage/'modal_state_by_era.csv',index=False);zero_share.to_csv(stage/'baseline_zero_share_by_era.csv',index=False);id_summary.to_csv(stage/'state_id_semantic_gate.csv',index=False);top_support.to_csv(stage/'topology_target_support.csv',index=False);gate.to_csv(stage/'topology_label_support_gate.csv',index=False)
  decision={'state_zero_semantic_pass':semantic_pass,'topology_label_support_pass':bool(label_pass),'state_anchor_identity_match_fraction':state_match,'model_trained':False,'reason':'No supervised subtype classifier is permitted: state semantic alignment and/or target support failed before any causal predictor can be fit.','next_action':'Stop subtype work. Keep binary trajectory transition-versus-stable context separate; do not collapse topology labels or relax support using 2026.'};dump(stage/'decision.json',decision)
  pd.DataFrame([{'table':'hourly_state_semantics','rows':len(hourly),'non_hourly_rows':0,'cadence':'1h'},{'table':'transition_topology_events','rows':len(events),'non_hourly_rows':0,'cadence':'1h'}]).to_csv(stage/'cadence_audit.csv',index=False)
  contract={'labels':'ex-post topology targets only; destination/type never used as a causal feature','semantic_precondition':'state 0 must be modal baseline in each 2022-25 era and zero-vs-nonzero contrast plus nonzero state profiles must be stable before supervised labels are admissible','support_precondition':'each required topology target requires >=12 events and >=3 pre-2026 eras; abstain/other is never folded into a named type','split':'only 2022-25 may establish labels/train; 2026 is held untouched and is not used to repair support','outcomes':'no economics, alpha, residual, policy, PnL or 1m path is read','cadence':'all rows are 1h'};dump(stage/'contract.json',contract)
  files=[p for p in stage.iterdir() if p.is_file()];status='SEALED_SUPERVISED_TAXONOMY_ADMISSIBLE' if semantic_pass and label_pass else 'SEALED_NEGATIVE_SUPERVISED_TAXONOMY_PRECONDITION_FAILURE';m={'schema':'supervised_coarse_topology_taxonomy_audit_v1','status':status,'promotion_eligible':False,'decision':decision,'contract':contract,'inputs_sha256':{str((LEDGER/'hourly_state_calendar.parquet').resolve()):sha(LEDGER/'hourly_state_calendar.parquet'),str((LEDGER/'transition_episode_ledger.parquet').resolve()):sha(LEDGER/'transition_episode_ledger.parquet'),str((CAT/'event_preonset_sequences.parquet').resolve()):sha(CAT/'event_preonset_sequences.parquet')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
