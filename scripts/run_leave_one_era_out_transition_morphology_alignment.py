#!/usr/bin/env python3
"""Train-only semantic alignment for leave-one-era-out transition morphology.

Each held era receives a fresh GMM fit only on the other eras.  A second,
independent train-only GMM supplies the semantic reference; Hungarian matching
maps the scoring model's components to that reference ordering.  Held-era rows
are never used to fit, order, align, calibrate, or define prototypes.

Outcome rows are post-event descriptive evidence only and remain grouped by
their supplied economic grade.  There is no baseline transition model on the
same event/outcome rows, so this runner cannot claim economic increment.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
CAT=ROOT/'data_perp/artifacts/transition_pattern_catalogue_20260730_v5'
OUTCOMES=ROOT/'data_perp/artifacts/transition_event_outcome_binding_20260730_v1/event_outcomes.parquet'
OUT=ROOT/'data_perp/artifacts/leave_one_era_out_transition_morphology_alignment_20260730_v1'
SCHEMA='leave_one_era_out_transition_morphology_alignment_v1'
N_COMPONENTS=3; MAX_FEATURES=64; MIN_EVENTS=8; MIN_HELD_ERAS=3; MIN_MAPPING_CONFIDENCE=.70; MIN_POSTERIOR_CORR=.70; MIN_PROTOTYPE_CORR=.70

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def safe(x:Any)->Any:
 if isinstance(x,(Path,pd.Timestamp)):return str(x)
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def wj(p:Path,x:Any)->None:
 t=p.with_name(f'.{p.name}.{os.getpid()}.partial');t.write_text(json.dumps(safe(x),indent=2,sort_keys=True)+'\n');os.replace(t,p)

def semantic_feature_columns(frame:pd.DataFrame,max_features:int=MAX_FEATURES)->list[str]:
 candidates=[c for c in frame if c.startswith('sequence__') and pd.api.types.is_numeric_dtype(frame[c]) and not c.endswith('complete_1h')]
 # Schema-order cap is deterministic and outcome-free.  Training eras then
 # decide which of these have sufficient finite support in the current fold.
 return candidates[:max_features]

def _corr(a:np.ndarray,b:np.ndarray)->float:
 if np.std(a)<=1e-12 or np.std(b)<=1e-12:return 1.0 if np.allclose(a,b) else 0.0
 return float(np.corrcoef(a,b)[0,1])

def hungarian_alignment(reference:np.ndarray,candidate:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
 """Return candidate-index per reference slot, costs and bounded confidence."""
 if reference.shape!=candidate.shape or reference.ndim!=2:raise ValueError('prototype matrices must share [component,descriptor] shape')
 scale=np.maximum(np.std(reference,axis=0),1e-6)
 cost=np.sqrt(np.mean(((reference[:,None,:]-candidate[None,:,:])/scale)**2,axis=2))
 r,c=linear_sum_assignment(cost)
 mapping=np.full(len(reference),-1,dtype=int);mapping[r]=c
 selected=cost[np.arange(len(reference)),mapping]
 confidence=np.exp(-selected)
 return mapping,selected,confidence

def _posterior_correlation(reference:np.ndarray,candidate:np.ndarray,mapping:np.ndarray)->np.ndarray:
 return np.asarray([_corr(reference[:,slot],candidate[:,mapping[slot]]) for slot in range(len(mapping))],dtype=float)

def _fit(values:np.ndarray,seed:int)->GaussianMixture:
 return GaussianMixture(n_components=N_COMPONENTS,covariance_type='diag',reg_covar=1e-5,n_init=5,random_state=seed).fit(values)

def align_leave_era_out(events:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
 data=events.copy();data['anchor_source_utc']=pd.to_datetime(data.anchor_source_utc,utc=True,errors='raise');data['era']=data.anchor_source_utc.dt.year.astype(str)
 if len(data)!=157 or data.event_id.duplicated().any():raise ValueError('requires the frozen 157-event unique catalogue')
 candidates=semantic_feature_columns(data); assignments=[];mappings=[];prototypes=[];folds=[]
 for held in sorted(data.era.unique()):
  train=data.loc[data.era.ne(held)].copy();test=data.loc[data.era.eq(held)].copy()
  cols=[c for c in candidates if train[c].notna().mean()>=.90 and train[c].std(skipna=True)>1e-10]
  if len(train)<N_COMPONENTS*8 or len(test)<1 or len(cols)<8:
   folds.append({'heldout_era':held,'status':'insufficient_train_or_feature_support','train_events':len(train),'held_events':len(test),'features':len(cols)});continue
  imp=SimpleImputer(strategy='median');scaler=StandardScaler(); a=scaler.fit_transform(imp.fit_transform(train[cols]));b=scaler.transform(imp.transform(test[cols]))
  reference=_fit(a,1729);candidate=_fit(a,481+int(held))
  # Descriptors are train-only component centroids in the fold's frozen,
  # standardized causal feature coordinate.  Reference ordering is itself
  # train-only and deterministic before matching.
  descriptor_count=min(12,len(cols));descriptor_idx=np.arange(descriptor_count)
  reference_desc=reference.means_[:,descriptor_idx];candidate_desc=candidate.means_[:,descriptor_idx]
  order=np.lexsort(tuple(reference_desc[:,i] for i in range(descriptor_count-1,-1,-1)))
  reference_desc=reference_desc[order]
  reference_train=reference.predict_proba(a)[:,order];candidate_train=candidate.predict_proba(a)
  mapping,cost,confidence=hungarian_alignment(reference_desc,candidate_desc)
  corr=_posterior_correlation(reference_train,candidate_train,mapping)
  candidate_test=candidate.predict_proba(b)[:,mapping]
  entropy=-(np.clip(candidate_test,1e-12,1)*np.log(np.clip(candidate_test,1e-12,1))).sum(axis=1)/np.log(N_COMPONENTS)
  ordered=np.sort(candidate_test,axis=1);margin=ordered[:,-1]-ordered[:,-2];slot=candidate_test.argmax(1)
  train_slot=reference_train.argmax(1)
  q01=float(np.quantile(reference.score_samples(a),.01)); held_log=reference.score_samples(b); ood=held_log<q01
  support=[]
  for semantic_slot in range(N_COMPONENTS):
   local=train.loc[train_slot==semantic_slot]
   support.append((int(len(local)),int(local.era.nunique())))
   prototypes.append({'heldout_era':held,'semantic_slot':semantic_slot,'train_events':len(train),'descriptor_features':'|'.join(cols[:descriptor_count]),'descriptor':json.dumps(reference_desc[semantic_slot].round(8).tolist()),'train_component_events':len(local),'train_component_eras':int(local.era.nunique())})
   mappings.append({'heldout_era':held,'semantic_slot':semantic_slot,'candidate_component':int(mapping[semantic_slot]),'hungarian_distance':float(cost[semantic_slot]),'mapping_confidence':float(confidence[semantic_slot]),'train_posterior_correlation':float(corr[semantic_slot]),'train_component_events':int(len(local)),'train_component_eras':int(local.era.nunique())})
  abstain=(candidate_test.max(axis=1)<.60)|(margin<.15)|ood|(confidence[slot]<MIN_MAPPING_CONFIDENCE)|(corr[slot]<MIN_POSTERIOR_CORR)
  out=test[['event_id','anchor_source_utc','era','source_state','destination_state']].copy();out['heldout_era']=held;out['semantic_slot']=slot;out['posterior_max']=candidate_test.max(axis=1);out['entropy']=entropy;out['top2_margin']=margin;out['ood']=ood;out['reference_log_density']=held_log;out['mapping_confidence']=[confidence[v] for v in slot];out['posterior_correlation']=[corr[v] for v in slot];out['train_component_events']=[support[v][0] for v in slot];out['train_component_eras']=[support[v][1] for v in slot];out['train_support_pass']=(out.train_component_events>=MIN_EVENTS)&(out.train_component_eras>=2);out['abstained']=abstain;out['semantic_component_id']=np.where(abstain,'abstain',['semantic_m%02d'%v for v in slot]);assignments.append(out)
  folds.append({'heldout_era':held,'status':'complete','train_events':len(train),'held_events':len(test),'features':len(cols),'train_score_q01':q01,'abstention_rate':float(abstain.mean()),'mean_mapping_confidence':float(confidence.mean()),'mean_train_posterior_correlation':float(corr.mean())})
 return pd.concat(assignments,ignore_index=True),pd.DataFrame(mappings),pd.DataFrame(prototypes),pd.DataFrame(folds)

def prototype_stability(prototypes:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for slot,g in prototypes.groupby('semantic_slot',sort=True):
  vector=np.asarray([json.loads(x) for x in g.descriptor],float); correlations=[]
  for i in range(len(vector)):
   for j in range(i+1,len(vector)):correlations.append(_corr(vector[i],vector[j]))
  rows.append({'semantic_slot':int(slot),'reference_folds':len(g),'prototype_pairwise_correlation_mean':float(np.mean(correlations)) if correlations else np.nan,'prototype_pairwise_correlation_min':float(np.min(correlations)) if correlations else np.nan,'prototype_stable':bool(correlations and min(correlations)>=MIN_PROTOTYPE_CORR)})
 return pd.DataFrame(rows)

def recurrence(assignments:pd.DataFrame, stability:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for slot,g in assignments.loc[assignments.semantic_slot.notna()].groupby('semantic_slot',sort=True):
  active=g.loc[~g.abstained]
  held_eras=int(active.heldout_era.nunique());events=len(active);stable=bool(stability.loc[stability.semantic_slot.eq(slot),'prototype_stable'].iloc[0])
  support=bool(held_eras>=MIN_HELD_ERAS and events>=MIN_EVENTS*MIN_HELD_ERAS)
  rows.append({'semantic_slot':int(slot),'heldout_eras':held_eras,'heldout_events':events,'abstention_rate':float(g.abstained.mean()),'mapping_confidence_mean':float(g.mapping_confidence.mean()),'posterior_correlation_mean':float(g.posterior_correlation.mean()),'held_support_pass':support,'prototype_stable':stable,'semantic_global_type_pass':bool(support and stable and (active.mapping_confidence>=MIN_MAPPING_CONFIDENCE).all() and (active.posterior_correlation>=MIN_POSTERIOR_CORR).all()),'reason':'requires recurrence, train-only alignment confidence/correlation and cross-fold prototype stability'})
 return pd.DataFrame(rows)

def outcome_increment(assignments:pd.DataFrame,outcomes:pd.DataFrame)->pd.DataFrame:
 """Grade-separated descriptive outcome slices; no baseline means no increment claim."""
 merged=assignments.merge(outcomes,on='event_id',how='inner',validate='one_to_one')
 rows=[]
 for (grade,era,component),g in merged.groupby(['source_grade','heldout_era','semantic_component_id'],sort=True):
  rows.append({'source_grade':grade,'heldout_era':era,'semantic_component_id':component,'events':len(g),'candidate_rows':int(g.candidate_rows.sum()),'mean_net_return':float(g.execution_net_ev_12h.mean()),'mean_gross_return':float(g.execution_gross_ev_12h.mean()),'outcome_increment_status':'NOT_IDENTIFIABLE_NO_MATCHED_CAUSAL_BASELINE','economic_pooling_forbidden':True})
 return pd.DataFrame(rows)

def run(output:Path=OUT)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 stage=output.with_name(f'.{output.name}.{os.getpid()}.partial');stage.mkdir(parents=True)
 try:
  events=pd.read_parquet(CAT/'event_preonset_sequences.parquet');outcomes=pd.read_parquet(OUTCOMES)
  assignments,mapping,prototypes,folds=align_leave_era_out(events);stability=prototype_stability(prototypes);support=recurrence(assignments,stability);economics=outcome_increment(assignments,outcomes)
  for n,t in {'aligned_oof_assignments.parquet':assignments,'fold_component_alignment.csv':mapping,'train_only_reference_prototypes.csv':prototypes,'fold_summary.csv':folds,'cross_fold_prototype_stability.csv':stability,'held_era_recurrence_support.csv':support,'outcome_increment_by_grade.csv':economics}.items():
   (t.to_parquet(stage/n,index=False) if n.endswith('.parquet') else t.to_csv(stage/n,index=False))
  global_pass=bool(len(support) and support.semantic_global_type_pass.all());status='SEALED_GLOBAL_TYPES_STILL_FAIL' if not global_pass else 'SEALED_TYPES_STABLE_BUT_ECONOMIC_INCREMENT_UNIDENTIFIABLE'
  report={'schema':SCHEMA,'status':status,'promotion_eligible':False,'events':157,'result':'No global morphology type is promotable: economic increment is not identifiable without a matched causal baseline, and semantic recurrence must remain diagnostic.' if not global_pass else 'Semantic recurrence passes its unsupervised gate, but no economic increment/policy claim is permitted.', 'contracts':{'fold':'leave one calendar era out','prototype':'reference and candidate GMM descriptors are fit/order/matched only on that fold training eras','alignment':'Hungarian matching; confidence and train posterior correlation emitted per semantic slot','held_era':'only transform/predict against frozen train-fold models; OOD and abstention fail closed','economics':'post-event descriptive only, separated by source_grade; no cross-grade pooling and no causal baseline'},'outputs_sha256':{p.name:sha(p) for p in stage.iterdir() if p.is_file()}}
  wj(stage/'report.json',report);manifest={'schema':SCHEMA+'_manifest','status':status,'runner_sha256':sha(Path(__file__)),'inputs_sha256':{str(CAT/'event_preonset_sequences.parquet'):sha(CAT/'event_preonset_sequences.parquet'),str(OUTCOMES):sha(OUTCOMES)},'outputs_sha256':{p.name:sha(p) for p in stage.iterdir() if p.is_file()}};wj(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');stage.replace(output);return manifest
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();print(json.dumps(safe(run(a.output)),indent=2))
