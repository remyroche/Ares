#!/usr/bin/env python3
"""Materialize all-hourly causal trajectory transition probabilities.

The classifier contract is the transfer-winning ``trajectory_only`` arm from
the train-only recurring-transition study.  It is deliberately distinct from
unstable transition prototype IDs: no component or destination identity enters
this sidecar.  Pre-2026 rows are calendar-era-held blocked OOF predictions;
2026 is scored by the frozen all-2022--2025 fit.  All rows are hourly and no
execution/trading outcome or one-minute path is read.
"""
from __future__ import annotations
import hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1'
CAT=ROOT/'data_perp/artifacts/transition_pattern_catalogue_20260730_v6'
STUDY=ROOT/'data_perp/artifacts/trainonly_recurring_transition_prototype_study_20260730_v3'
OUT=ROOT/'data_perp/artifacts/hourly_trajectory_transition_soft_sidecar_20260730_v1'
SPLIT=pd.Timestamp('2026-01-01',tz='UTC')
import sys
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_trainonly_recurring_transition_prototype_study import BASE_SIGNALS,PHASES,make_classifier

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p:Path,x:object)->None:
 q=p.with_name('.'+p.name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def hourly(t:pd.Series,name:str)->None:
 if t.duplicated().any() or (t.astype('int64')%pd.Timedelta(hours=1).value!=0).any():raise ValueError(f'{name} must be unique hourly')
def entropy(p:np.ndarray)->np.ndarray:return -(p*np.log(np.clip(p,1e-12,1))+(1-p)*np.log(np.clip(1-p,1e-12,1)))
def ece(y:np.ndarray,p:np.ndarray,bins:int=10)->float:
 edges=np.linspace(0,1,bins+1);tot=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
  if m.any():tot+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(tot)

def fields()->list[str]:
 out=[]
 for _,(h,stats) in PHASES.items():
  for signal in BASE_SIGNALS:
   for stat in stats:out.append(f'sequence__{signal}__{stat}_{h}h')
 return out

def trailing_features(panel:pd.DataFrame, wanted:list[str])->pd.DataFrame:
 """Exact [anchor-H, anchor) summary semantics used by anchor training."""
 x=panel.copy();x['source_utc']=pd.to_datetime(x.source_utc,utc=True,errors='raise');hourly(x.source_utc,'hourly panel')
 segment='source_segment_id' if 'source_segment_id' in x else 'calendar_segment_id'
 x=x.sort_values([segment,'source_utc'],kind='stable').reset_index(drop=True)
 # A gap starts a new effective segment; no rolling path may cross it.
 x['_contiguous_block']=x.groupby(segment,observed=True).source_utc.diff().ne(pd.Timedelta(hours=1)).groupby(x[segment]).cumsum().astype(str)
 x['_trajectory_block']=x[segment].astype(str)+'|'+x['_contiguous_block']
 produced=pd.DataFrame({'source_utc':x.source_utc,'source_state':pd.to_numeric(x['target__pooled_state'],errors='coerce')})
 for signal in BASE_SIGNALS:
  if signal not in x:raise KeyError(signal)
  values=pd.to_numeric(x[signal],errors='coerce')
  for _,(h,stats) in PHASES.items():
   prior=values.groupby(x['_trajectory_block'],observed=True).shift(1)
   group=prior.groupby(x['_trajectory_block'],observed=True)
   valid=group.rolling(h,min_periods=h).count().reset_index(level=0,drop=True).eq(h)
   if 'mean' in stats:produced[f'sequence__{signal}__mean_{h}h']=group.rolling(h,min_periods=h).mean().reset_index(level=0,drop=True)
   if 'delta' in stats:produced[f'sequence__{signal}__delta_{h}h']=(prior-group.shift(h-1)).where(valid)
   if 'slope_per_hour' in stats:
    weights=np.arange(h,dtype=float);weights-=weights.mean();den=float((weights**2).sum())
    # full sequence is required; ``min_periods`` makes any missing raw point
    # unavailable exactly as the original causal summarizer does.
    produced[f'sequence__{signal}__slope_per_hour_{h}h']=group.rolling(h,min_periods=h).apply(lambda a:float(np.dot(weights,a)/den),raw=True).reset_index(level=0,drop=True).where(valid)
 missing=set(wanted).difference(produced)
 if missing:raise KeyError(sorted(missing))
 return produced

def validate_anchor_equivalence(features:pd.DataFrame,wanted:list[str])->dict:
 anchor=pd.read_parquet(CAT/'stable_transition_sequence_inputs.parquet')
 anchor['anchor_source_utc']=pd.to_datetime(anchor.anchor_source_utc,utc=True)
 joined=anchor[['event_id','anchor_source_utc',*wanted]].merge(features[['source_utc',*wanted]],left_on='anchor_source_utc',right_on='source_utc',how='left',validate='one_to_one',suffixes=('_anchor','_hourly'))
 if len(joined)!=len(anchor):raise ValueError('anchor join failed')
 errors=[]
 for f in wanted:
  a=pd.to_numeric(joined[f+'_anchor'],errors='coerce').to_numpy(float);b=pd.to_numeric(joined[f+'_hourly'],errors='coerce').to_numpy(float)
  same=np.isclose(a,b,rtol=1e-6,atol=1e-7,equal_nan=True)
  errors.append((f,int((~same).sum()),float(np.nanmax(np.abs(a-b)) if np.isfinite(np.abs(a-b)).any() else 0)))
 failures=[z for z in errors if z[1]]
 if failures:raise ValueError(f'not feature-identical to anchor training: {failures[:3]}')
 return {'anchor_rows':len(anchor),'fields':len(wanted),'max_abs_error':max(z[2] for z in errors),'mismatch_fields':0}

def load_anchors(wanted:list[str])->pd.DataFrame:
 x=pd.read_parquet(CAT/'stable_transition_sequence_inputs.parquet');x['anchor_source_utc']=pd.to_datetime(x.anchor_source_utc,utc=True);x['era']=x.anchor_source_utc.dt.year.astype(int)
 if x.sequence_available_utc.pipe(pd.to_datetime,utc=True).gt(x.anchor_source_utc).any():raise ValueError('noncausal anchor sequence')
 if set(x.target__stable_vs_transition.unique())!={0,1}:raise ValueError('labels')
 return x

def run(output:Path=OUT)->Path:
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 manifest=json.loads((STUDY/'manifest.json').read_text())
 if manifest['status']!='SEALED_NO_STABLE_RECURRING_PROTOTYPES':raise ValueError('expected sealed study')
 wanted=fields();panel=pd.read_parquet(LEDGER/'hourly_state_calendar.parquet');feat=trailing_features(panel,wanted);eq=validate_anchor_equivalence(feat,wanted)
 anchors=load_anchors(wanted);train=anchors[anchors.anchor_source_utc<SPLIT].copy();assess=anchors[anchors.anchor_source_utc>=SPLIT].copy()
 if train.era.max()!=2025 or assess.era.min()!=2026:raise ValueError('split')
 work=feat.copy();work['era']=work.source_utc.dt.year.astype(int);work['trajectory_available']=work[wanted+['source_state']].notna().all(axis=1)
 work['trajectory_transition_probability']=np.nan;work['probability_entropy']=np.nan;work['top2_margin']=np.nan;work['oof_held_era']=pd.Series(pd.NA,index=work.index,dtype='Int64');work['provenance_partition']='unavailable_warmup_or_missing_lookback';work['fit_train_eras']=''
 calibration=[]
 for era in sorted(train.era.unique()):
  fit=train[train.era.ne(era)];model=make_classifier(wanted);model.fit(fit[wanted+['source_state']],fit.target__stable_vs_transition)
  m=work.era.eq(era)&work.trajectory_available
  p=model.predict_proba(work.loc[m,wanted+['source_state']])[:,1];work.loc[m,'trajectory_transition_probability']=p;work.loc[m,'probability_entropy']=entropy(p);work.loc[m,'top2_margin']=np.abs(2*p-1);work.loc[m,'oof_held_era']=era;work.loc[m,'provenance_partition']='blocked_era_oof';work.loc[m,'fit_train_eras']=','.join(str(v) for v in sorted(set(train.era)-{era}))
  held=train[train.era.eq(era)];row=held[['event_id','anchor_source_utc','target__stable_vs_transition']].merge(work[['source_utc','trajectory_transition_probability']],left_on='anchor_source_utc',right_on='source_utc',how='left',validate='one_to_one');row=row.loc[row.trajectory_transition_probability.notna()];y=row.target__stable_vs_transition.to_numpy(int);q=row.trajectory_transition_probability.to_numpy(float)
  calibration.append({'partition':'blocked_era_oof_anchor','held_era':era,'rows':len(row),'transition_events':int(y.sum()),'roc_auc':float(roc_auc_score(y,q)),'average_precision':float(average_precision_score(y,q)),'brier':float(brier_score_loss(y,q)),'ece10':ece(y,q)})
 model=make_classifier(wanted);model.fit(train[wanted+['source_state']],train.target__stable_vs_transition)
 m=work.era.ge(2026)&work.trajectory_available;p=model.predict_proba(work.loc[m,wanted+['source_state']])[:,1];work.loc[m,'trajectory_transition_probability']=p;work.loc[m,'probability_entropy']=entropy(p);work.loc[m,'top2_margin']=np.abs(2*p-1);work.loc[m,'provenance_partition']='untouched_2026_frozen_fit';work.loc[m,'fit_train_eras']='2022,2023,2024,2025'
 held=assess[['event_id','anchor_source_utc','target__stable_vs_transition']].merge(work[['source_utc','trajectory_transition_probability']],left_on='anchor_source_utc',right_on='source_utc',how='left',validate='one_to_one');held=held.loc[held.trajectory_transition_probability.notna()];y=held.target__stable_vs_transition.to_numpy(int);q=held.trajectory_transition_probability.to_numpy(float);calibration.append({'partition':'untouched_2026_anchor','held_era':pd.NA,'rows':len(held),'transition_events':int(y.sum()),'roc_auc':float(roc_auc_score(y,q)),'average_precision':float(average_precision_score(y,q)),'brier':float(brier_score_loss(y,q)),'ece10':ece(y,q)})
 if not work.loc[work.era.lt(2026)&work.trajectory_available,'provenance_partition'].eq('blocked_era_oof').all():raise ValueError('pre2026 oof coverage')
 if not work.loc[work.era.ge(2026)&work.trajectory_available,'provenance_partition'].eq('untouched_2026_frozen_fit').all():raise ValueError('2026 frozen coverage')
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  columns=['source_utc','era','source_state','trajectory_available','trajectory_transition_probability','probability_entropy','top2_margin','oof_held_era','provenance_partition','fit_train_eras']
  work[columns].to_parquet(stage/'hourly_trajectory_transition_soft_sidecar.parquet',index=False)
  pd.DataFrame(calibration).to_csv(stage/'anchor_calibration.csv',index=False)
  pd.DataFrame([{'table':'hourly_trajectory_transition_soft_sidecar','rows':len(work),'non_hourly_rows':0,'duplicate_timestamp_rows':0,'model_sample_cadence':'1h','assessment_sample_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only'}]).to_csv(stage/'cadence_audit.csv',index=False)
  pd.DataFrame([{'era':int(e),'rows':int(len(z)),'available_rows':int(z.trajectory_available.sum()),'partition':str(z.provenance_partition.mode().iloc[0])} for e,z in work.groupby('era',sort=True)]).to_csv(stage/'coverage_by_era.csv',index=False)
  contract={'model':'trajectory_only logistic transition-vs-stable classifier from trainonly recurrence study; source_state plus fixed causal 168/24/6/3h trajectory features, no current sidecar fields because that arm transferred worse','fit':'pre2026 rows use leave-calendar-era-out blocked OOF models (not a live chronological fit); 2026 uses one all-2022-2025 frozen model','no_clusters':'no prototype ID, destination state, topology, execution outcome, trading score, PnL or 1m path is emitted/read','identity_uncertainty':'source_utc is one hourly identity; probability entropy/top2 margin and held-era/provenance are emitted','integration':'eligible as a diagnostic OOF/frozen soft context candidate for a separately specified GAM/base-residual ablation; no direct policy/gate authority'};dump(stage/'contract.json',contract);dump(stage/'anchor_feature_equivalence.json',eq)
  files=[p for p in stage.iterdir() if p.is_file()];m={'schema':'hourly_trajectory_transition_soft_sidecar_v1','status':'SEALED_HOURLY_OOF_AND_FROZEN_TRAJECTORY_CONTEXT_NON_PROMOTION','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','counts':{'hourly_rows':len(work),'pre2026_available_rows':int((work.era.lt(2026)&work.trajectory_available).sum()),'untouched_2026_available_rows':int((work.era.ge(2026)&work.trajectory_available).sum()),'anchor_train_rows':len(train),'anchor_assessment_rows':len(assess)},'contract':contract,'inputs_sha256':{str((LEDGER/'hourly_state_calendar.parquet').resolve()):sha(LEDGER/'hourly_state_calendar.parquet'),str((CAT/'stable_transition_sequence_inputs.parquet').resolve()):sha(CAT/'stable_transition_sequence_inputs.parquet'),str((STUDY/'manifest.json').resolve()):sha(STUDY/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
