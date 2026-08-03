#!/usr/bin/env python3
"""Robust, abstaining coarse transition taxonomy: train 2022--25, assess 2026.

This is intentionally not an unconstrained high-dimensional clustering run.
It has fixed causal phase/horizon semantic profiles, train-only winsorisation,
an explicit robust-distance outlier bucket, only K=2/3 core candidates, and
support/bootstrap/leave-era/2026-transfer evidence.  Destination state is only
reported as post-labelled topology; no trading/1m outcome is read.
"""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score

ROOT=Path(__file__).resolve().parents[1]
CAT=ROOT/'data_perp/artifacts/transition_pattern_catalogue_20260730_v6'
BINARY=ROOT/'data_perp/artifacts/trainonly_recurring_transition_prototype_study_20260730_v3'
OUT=ROOT/'data_perp/artifacts/constrained_coarse_transition_taxonomy_20260730_v3'
SPLIT=pd.Timestamp('2026-01-01',tz='UTC');SEED=730
GROUPS={
 'breadth':('breadth_dispersion','downside_breadth_intensity','btc_resilience_alt_weakness'),
 'washout_reversal':('broad_washout_recovery','deleveraged_range_climax_reversal','deleveraging_without_followthrough','short_breakout_exhaustion'),
 'funding_positioning':('funding_confirmed_long_flush','funding_confirmed_short_covering','funding_deleveraging_divergence'),
 'correlation_dispersion':('correlation_breakdown_dispersion','peer_volatility_decoupling'),
}
PHASES={'precondition_168h':(168,('mean','delta')),'approach_24h':(24,('mean','delta')),'acceleration_6h':(6,('slope_per_hour','delta')),'trigger_3h':(3,('slope_per_hour','delta'))}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p:Path,x:object):
 q=p.with_name('.'+p.name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)

def feature_spec(frame:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for group,signals in GROUPS.items():
  for phase,(h,stats) in PHASES.items():
   for signal in signals:
    for stat in stats:
     f=f'sequence__{signal}__{stat}_{h}h'
     if f not in frame:raise KeyError(f)
     rows.append({'group':group,'phase':phase,'horizon_hours':h,'statistic':stat,'signal':signal,'field':f,'causal_preonset':True})
 return pd.DataFrame(rows)

class Profile:
 def __init__(self,spec:pd.DataFrame):self.spec=spec.reset_index(drop=True)
 def fit(self,x:pd.DataFrame):
  a=x[self.spec.field.tolist()].apply(pd.to_numeric,errors='coerce').to_numpy(float)
  self.impute=np.nanmedian(a,axis=0);a=np.where(np.isnan(a),self.impute,a)
  self.lo=np.quantile(a,.05,axis=0);self.hi=np.quantile(a,.95,axis=0);a=np.clip(a,self.lo,self.hi)
  self.med=np.median(a,axis=0);q=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0)
  # Sparse causal indicators can have a tiny nonzero IQR.  Treating that as a
  # scale would manufacture enormous pseudo-distances; below 1e-2 retain raw
  # clipped units instead.  This threshold is fixed before any 2026 view.
  self.scale=np.where(q>.01,q,1.)
  self.columns=[f'profile__{g}__{p}' for g in GROUPS for p in PHASES]
  return self
 def transform(self,x:pd.DataFrame)->np.ndarray:
  a=x[self.spec.field.tolist()].apply(pd.to_numeric,errors='coerce').to_numpy(float);a=np.where(np.isnan(a),self.impute,a);a=(np.clip(a,self.lo,self.hi)-self.med)/self.scale
  values=[]
  for group in GROUPS:
   for phase in PHASES:
    idx=np.flatnonzero((self.spec.group.eq(group)&self.spec.phase.eq(phase)).to_numpy())
    # Full production contracts always cover each fixed family/phase.  A zero
    # placeholder keeps focused unit fixtures well-defined without changing
    # the production feature schema.
    values.append(a[:,idx].mean(axis=1) if len(idx) else np.zeros(len(x)))
  return np.column_stack(values)

def fit_taxonomy(x:pd.DataFrame,spec:pd.DataFrame,k:int):
 profile=Profile(spec).fit(x);z=profile.transform(x);center=np.median(z,axis=0);mad=np.median(np.abs(z-center),axis=0);mad=np.where(mad>1e-9,mad,1.)
 distance=np.sqrt((((z-center)/mad)**2).mean(axis=1));threshold=float(np.quantile(distance,.975));core=distance<=threshold
 if core.sum()<k*12:raise ValueError('insufficient inlier support')
 model=KMeans(n_clusters=k,n_init=100,random_state=SEED+k).fit(z[core]);labels=np.full(len(x),-1,dtype=int);labels[core]=model.predict(z[core])
 return profile,model,center,mad,threshold,z,distance,labels

def predict_taxonomy(x:pd.DataFrame,fit:tuple):
 profile,model,center,mad,threshold,*_=fit;z=profile.transform(x);distance=np.sqrt((((z-center)/mad)**2).mean(axis=1));labels=np.full(len(x),-1,dtype=int);core=distance<=threshold;labels[core]=model.predict(z[core]);d2=((z[:,None,:]-model.cluster_centers_[None,:,:])**2).sum(axis=2);w=np.exp(-.5*(d2-d2.min(axis=1,keepdims=True)));prob=w/w.sum(axis=1,keepdims=True);return z,distance,labels,prob

def support(x:pd.DataFrame,labels:np.ndarray)->pd.DataFrame:
 z=x.assign(component=labels);return z[z.component.ge(0)].groupby('component',observed=True).agg(events=('event_id','size'),eras=('era','nunique')).reset_index()

def bootstrap(train:pd.DataFrame,spec:pd.DataFrame,k:int,ref_labels:np.ndarray,ref_core:np.ndarray,draws:int=120):
 rng=np.random.default_rng(SEED+k);aris=[]
 for _ in range(draws):
  sample=train.iloc[rng.integers(0,len(train),len(train))]
  try:
   fit=fit_taxonomy(sample,spec,k);_,_,label,_=predict_taxonomy(train,fit);common=ref_core&(label>=0)
   if common.sum()>=20:aris.append(adjusted_rand_score(ref_labels[common],label[common]))
  except ValueError:pass
 return {'bootstrap_draws':draws,'bootstrap_valid_draws':len(aris),'bootstrap_ari_mean':float(np.mean(aris)) if aris else np.nan,'bootstrap_ari_q05':float(np.quantile(aris,.05)) if aris else np.nan,'bootstrap_ari_q95':float(np.quantile(aris,.95)) if aris else np.nan}

def align(a:np.ndarray,b:np.ndarray):
 aa=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12);bb=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12);sim=aa@bb.T;i,j=linear_sum_assignment(-sim);return float(sim[i,j].mean()),float(sim[i,j].min()),json.dumps([(int(x),int(y),float(sim[x,y])) for x,y in zip(i,j)])

def leave_era(train:pd.DataFrame,spec:pd.DataFrame,k:int):
 fits={};assign=[]
 for era in sorted(train.era.unique()):
  fit=fit_taxonomy(train[train.era.ne(era)],spec,k);z,d,l,p=predict_taxonomy(train[train.era.eq(era)],fit)
  fits[int(era)]=fit[1].cluster_centers_
  held=train[train.era.eq(era)]
  assign.extend({'held_era':int(era),'event_id':e,'component':int(c),'outlier_unclassified':bool(c<0),'robust_distance':float(dd),'component_confidence':float(pp.max()) if c>=0 else np.nan} for e,c,dd,pp in zip(held.event_id,l,d,p))
 rows=[]
 for left,right in combinations(sorted(fits),2):
  mean,minimum,matches=align(fits[left],fits[right]);rows.append({'fold_a':left,'fold_b':right,'mean_matched_semantic_cosine':mean,'min_matched_semantic_cosine':minimum,'matches':matches})
 return pd.DataFrame(rows),pd.DataFrame(assign)

def binary_metrics()->pd.DataFrame:
 return pd.read_csv(BINARY/'transition_vs_stable_transfer.csv').assign(source='separate_binary_transition_vs_stable_study')

def run(output:Path=OUT)->Path:
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 x=pd.read_parquet(CAT/'stable_transition_sequence_inputs.parquet');x['anchor_source_utc']=pd.to_datetime(x.anchor_source_utc,utc=True);x['sequence_available_utc']=pd.to_datetime(x.sequence_available_utc,utc=True)
 if x.event_id.duplicated().any() or not x.sequence_available_utc.le(x.anchor_source_utc).all() or (x.anchor_source_utc.astype('int64')%pd.Timedelta(hours=1).value!=0).any():raise ValueError('identity/cadence/causality')
 x['era']=x.anchor_source_utc.dt.year.astype(int);x['topology']=np.where(x.target__stable_vs_transition.eq(0),'stable','state_'+x.source_state.astype(str)+'_to_state_'+x.destination_state.astype(str))
 spec=feature_spec(x);events=x[x.target__stable_vs_transition.eq(1)].copy();train=events[events.anchor_source_utc<SPLIT].copy();assess=events[events.anchor_source_utc>=SPLIT].copy()
 if train.era.max()!=2025 or assess.era.min()!=2026:raise ValueError('split')
 candidates=[];fits={}
 for k in (2,3):
  fit=fit_taxonomy(train,spec,k);labels=fit[-1];sup=support(train,labels);core=labels>=0;boot=bootstrap(train,spec,k,labels,core);alignments,_=leave_era(train,spec,k)
  row={'k':k,'train_events':len(train),'outlier_unclassified_events':int((~core).sum()),'inlier_events':int(core.sum()),'silhouette_core':float(silhouette_score(fit[-3][core],labels[core])),'min_component_events':int(sup.events.min()),'min_component_eras':int(sup.eras.min()),'leave_era_min_cosine':float(alignments.min_matched_semantic_cosine.min()),'leave_era_mean_cosine':float(alignments.mean_matched_semantic_cosine.mean()),**boot}
  row['passes_gate']=bool(row['outlier_unclassified_events']<=max(8,round(.10*len(train))) and row['min_component_events']>=12 and row['min_component_eras']>=3 and row['bootstrap_ari_mean']>=.60 and row['bootstrap_ari_q05']>=.35 and row['leave_era_min_cosine']>=.50)
  candidates.append(row);fits[k]=(fit,alignments)
 candidates=pd.DataFrame(candidates);passed=candidates[candidates.passes_gate];selected=int(passed.sort_values(['bootstrap_ari_mean','leave_era_mean_cosine'],ascending=False).iloc[0].k) if len(passed) else None;diagnostic=int(candidates.sort_values(['bootstrap_ari_mean','leave_era_mean_cosine'],ascending=False).iloc[0].k)
 fit,alignments=fits[selected if selected else diagnostic];ztr,dtr,ltr,ptr=predict_taxonomy(train,fit);zte,dte,lte,pte=predict_taxonomy(assess,fit);_,held=leave_era(train,spec,selected if selected else diagnostic)
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  candidates.to_csv(stage/'coarse_taxonomy_candidate_gates.csv',index=False);spec.to_csv(stage/'phase_horizon_profile_contract.csv',index=False);alignments.to_csv(stage/'leave_era_semantic_alignment.csv',index=False);held.to_csv(stage/'leave_era_assignments.csv',index=False);binary_metrics().to_csv(stage/'transition_vs_stable_separate_metrics.csv',index=False)
  def membership(frame,labels,distance,prob):
   o=frame[['event_id','anchor_source_utc','era','source_state','destination_state','topology']].copy();o['diagnostic_component']=labels;o['outlier_unclassified']=labels<0;o['robust_distance']=distance;o['component_entropy']=-(prob*np.log(np.clip(prob,1e-12,1))).sum(axis=1);o['component_top2_margin']=np.sort(prob,axis=1)[:,-1]-np.sort(prob,axis=1)[:,-2];o['component_is_promotable_type']=selected is not None;return o
  a=membership(train,ltr,dtr,ptr);b=membership(assess,lte,dte,pte);a.to_parquet(stage/'train_2022_2025_membership.parquet',index=False);b.to_parquet(stage/'assessment_2026_membership.parquet',index=False)
  pd.concat([a.assign(partition='train_2022_2025'),b.assign(partition='assessment_2026')]).groupby(['partition','diagnostic_component','topology'],dropna=False,observed=True).size().rename('events').reset_index().to_csv(stage/'component_topology_support.csv',index=False)
  # Fixed profile medians make each proposed coarse type interpretable without
  # using state destination as a causal feature.
  profiles=pd.DataFrame(ztr,columns=fit[0].columns);profiles['diagnostic_component']=ltr;profiles['outlier_unclassified']=ltr<0;profiles.groupby(['diagnostic_component','outlier_unclassified'],dropna=False,observed=True).median().reset_index().to_csv(stage/'train_phase_horizon_profiles.csv',index=False)
  cadence=pd.DataFrame([{'table':'all_transition_stable_anchors','rows':len(x),'non_hourly_rows':0,'cadence':'1h'},{'table':'train_transition_events','rows':len(train),'non_hourly_rows':0,'cadence':'1h'},{'table':'untouched_2026_transition_events','rows':len(assess),'non_hourly_rows':0,'cadence':'1h'}]);cadence.to_csv(stage/'cadence_audit.csv',index=False)
  contract={'discovery':'only transition anchors from 2022-2025; K is restricted to 2/3 and selected only if all train-only gates pass','profiles':'fixed outcome-free pre-onset 168/24/6/3h semantic profile fields; train-only median imputation, 5/95% winsorisation and robust scaling','outlier':'distance above the train 97.5th percentile is an explicit unclassified bucket, never forced into a type','separation':'separate binary transition-vs-stable metrics are retained; source/destination topology is reported only after labels and destination never enters profile features','transfer':'2026 transforms the frozen selected (or diagnostic-only failure) train geometry; no 2026 fit/selection','cadence':'all model rows are 1h; no execution/trading outcome or 1m path is read'};dump(stage/'contract.json',contract)
  files=[p for p in stage.iterdir() if p.is_file()];status='SEALED_STABLE_COARSE_TYPES_DIAGNOSTIC_ONLY' if selected else 'SEALED_STRONGER_NEGATIVE_COARSE_TAXONOMY';decision=f'K={selected} passes all coarse taxonomy gates.' if selected else f'Neither K=2 nor K=3 clears robust outlier/support/bootstrap/leave-era gates; diagnostic K={diagnostic} is unclassified/local only.'
  m={'schema':'constrained_coarse_transition_taxonomy_v3','status':status,'promotion_eligible':False,'selected_k':selected,'diagnostic_k':diagnostic,'decision':decision,'counts':{'anchors':len(x),'train_transition_events':len(train),'assessment_transition_events':len(assess)},'contract':contract,'inputs_sha256':{str((CAT/'stable_transition_sequence_inputs.parquet').resolve()):sha(CAT/'stable_transition_sequence_inputs.parquet'),str((BINARY/'manifest.json').resolve()):sha(BINARY/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
