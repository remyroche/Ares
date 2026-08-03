#!/usr/bin/env python3
"""Diagnose v1/v2 transition non-transfer without fitting or selecting on 2026."""
from __future__ import annotations
import argparse,json,os,shutil,sys,uuid
from pathlib import Path
from typing import Any,Sequence
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_strict_forward_transition_evaluation import ART,CATALOGUE,label_available,safe,sha256
OUT=ART/'strict_transition_nontransfer_diagnostic_20260730_v2';V1=ART/'strict_forward_transition_evaluation_20260730_v1/forward_transition_predictions.parquet';V2=ART/'strict_forward_transition_challenger_20260730_v2/v2_forward_predictions.parquet';FEATURES=ART/'strict_forward_transition_challenger_20260730_v2/selected_features.json'
TRAIN_END=pd.Timestamp('2026-01-01',tz='UTC')
def zero_positive_status(g:pd.DataFrame, availability:pd.Series, catalogue_close:pd.Timestamp)->str:
 if g.target__transition_active.isna().any():return 'UNRESOLVED_OR_MISSING_ACTIVE_LABEL'
 if (availability.loc[g.index]>catalogue_close).any():return 'LABEL_AVAILABILITY_AFTER_CATALOGUE_HORIZON'
 return 'GENUINE_ZERO_ACTIVE_LABELS_IN_MATERIALIZED_CATALOGUE'
def run(*,catalogue:Path=CATALOGUE,v1:Path=V1,v2:Path=V2,features:Path=FEATURES,output:Path=OUT)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 f=pd.read_parquet(catalogue).copy();f.source_utc=pd.to_datetime(f.source_utc,utc=True);a=label_available(f);f['month']=f.source_utc.dt.strftime('%Y-%m');train=f.loc[f.source_utc.lt(TRAIN_END)&a.lt(TRAIN_END)].copy();test=f.loc[f.source_utc.ge(TRAIN_END)].copy();test=test.loc[test.source_utc.le(pd.to_datetime(pd.read_parquet(v2,columns=['source_utc']).source_utc.max(),utc=True))]
 monthly=[]
 catalogue_close=f.source_utc.max()+pd.Timedelta(hours=12)
 for month,g in test.groupby('month',sort=True):monthly.append({'month':month,'hours':len(g),'active_labels':int(g.target__transition_active.notna().sum()),'active_events':int(g.target__transition_active.fillna(0).sum()),'active_rate':g.target__transition_active.mean(),'pattern_phase_labels':int(g.target__pattern_phase.notna().sum()),'event_ids':int(g.target__event_id.notna().sum()),'active_label_available_max':a.loc[g.index].max(),'all_active_labels_available_by_catalogue_close':bool(a.loc[g.index].le(catalogue_close).all()),'zero_positive_status':zero_positive_status(g,a,catalogue_close) if g.target__transition_active.fillna(0).sum()==0 else 'HAS_ACTIVE_EVENTS'})
 monthly=pd.DataFrame(monthly)
 life=pd.concat([train.assign(split='train_2022_25'),test.assign(split='test_2026')]).groupby(['split','target__pattern_phase'],dropna=False).agg(hours=('source_utc','size'),active_rate=('target__transition_active','mean')).reset_index()
 names=json.loads(features.read_text());shift=[]
 for col in names:
  x=pd.to_numeric(train[col],errors='coerce').dropna();y=pd.to_numeric(test[col],errors='coerce').dropna()
  if len(x)>20 and len(y)>20:
   ks=ks_2samp(x,y);scale=x.std() or np.nan;shift.append({'feature':col,'train_n':len(x),'test_n':len(y),'train_mean':x.mean(),'test_mean':y.mean(),'standardized_mean_shift':(y.mean()-x.mean())/scale,'ks_stat':ks.statistic,'ks_pvalue':ks.pvalue})
 shift=pd.DataFrame(shift).sort_values('ks_stat',ascending=False)
 numeric_train=train[names].apply(pd.to_numeric,errors='coerce');numeric_test=test[names].apply(pd.to_numeric,errors='coerce');ct=numeric_train.corr();ce=numeric_test.corr();pairs=[]
 for i,left in enumerate(names):
  for right in names[i+1:]:
   if np.isfinite(ct.loc[left,right]) and np.isfinite(ce.loc[left,right]):pairs.append({'left_feature':left,'right_feature':right,'train_correlation':ct.loc[left,right],'test_correlation':ce.loc[left,right],'correlation_shift':ce.loc[left,right]-ct.loc[left,right]})
 cov=pd.DataFrame(pairs);cov['abs_shift']=cov.correlation_shift.abs();cov=cov.sort_values('abs_shift',ascending=False).head(100)
 one=pd.read_parquet(v1);two=pd.read_parquet(v2);one.source_utc=pd.to_datetime(one.source_utc,utc=True);two.source_utc=pd.to_datetime(two.source_utc,utc=True);score=one[['source_utc','target__transition_active','transition_probability']].merge(two[['source_utc','transition_probability']],on='source_utc',suffixes=('_v1','_v2'));score['month']=score.source_utc.dt.strftime('%Y-%m');score['rank_v1']=score.transition_probability_v1.rank(pct=True);score['rank_v2']=score.transition_probability_v2.rank(pct=True);score['decile_v1']=pd.qcut(score.rank_v1,10,labels=False,duplicates='drop');score['decile_v2']=pd.qcut(score.rank_v2,10,labels=False,duplicates='drop');cal=[]
 for arm in ('v1','v2'):
  for decile,g in score.groupby(f'decile_{arm}',sort=True):cal.append({'arm':arm,'scope':'all_2026','score_decile':decile,'hours':len(g),'mean_probability':g[f'transition_probability_{arm}'].mean(),'observed_active_rate':g.target__transition_active.mean()})
 agreement=score.groupby('month').agg(hours=('source_utc','size'),spearman_rank_agreement=('rank_v1',lambda x:x.corr(score.loc[x.index,'rank_v2'],method='spearman')),v1_mean_probability=('transition_probability_v1','mean'),v2_mean_probability=('transition_probability_v2','mean'),observed_active_rate=('target__transition_active','mean')).reset_index()
 morph=pd.concat([train.assign(split='train_2022_25'),test.assign(split='test_2026')]).groupby(['split','target__transition_archetype'],dropna=False).agg(hours=('source_utc','size'),active_rate=('target__transition_active','mean')).reset_index()
 todo=pd.DataFrame([{'id':'multi_horizon_onset_heads','pre_registered_selection':'Choose H1/H3/H6/H12 onset/active heads only by blocked 2022-25 AP-Brier and fold stability; all labels require their own resolved availability.','rationale':'2026 base-rate timing is intermittent; preserve horizons separately rather than collapsing to one active target.'},{'id':'competing_risk_lifecycle','pre_registered_selection':'Train cause-specific onset/approach/trigger/active/destination heads with a training-only multinomial/competing-risk loss; phase remains target only.','rationale':'Low macro-F1 and altered lifecycle support indicate a single phase classifier conflates rare paths.'},{'id':'train_only_rate_and_calibration','pre_registered_selection':'Compare prior-fold Platt, isotonic with minimum-support abstention, and no calibration using 2022-25 blocked folds only.','rationale':'v2 improves Brier/ECE but loses AP, so calibration and ranking should be selected separately.'},{'id':'robust_transition_features','pre_registered_selection':'From the current causal list, use only features stable across blocked 2022-25 eras (distribution/covariance stability) and test dynamic vs structural families in blocked CV.','rationale':'2026 shift diagnostics identify candidate drift; no 2026 feature is selected.'}])
 stage=output.parent/f'.{output.name}.{uuid.uuid4().hex}.stage';stage.mkdir(parents=True)
 try:
  monthly.to_csv(stage/'monthly_label_coverage_and_base_rates.csv',index=False);life.to_csv(stage/'lifecycle_class_support.csv',index=False);shift.to_csv(stage/'feature_distribution_shift.csv',index=False);cov.to_csv(stage/'feature_covariance_shift_top100.csv',index=False);pd.DataFrame(cal).to_csv(stage/'score_rank_calibration.csv',index=False);agreement.to_csv(stage/'v1_v2_score_agreement_by_month.csv',index=False);morph.to_csv(stage/'event_morphology_shift.csv',index=False);todo.to_csv(stage/'pre_registered_next_ablations.csv',index=False)
  manifest={'schema':'strict_transition_nontransfer_diagnostic_v2','research_only':True,'promotion_eligible':False,'contract':'descriptive 2026 diagnosis only; no retraining, feature/HPO selection, calibration fitting, or policy gate uses 2026','zero_positive_contract':'zero-positive month is genuine only if every active label is materialized/non-null and all its declared/fallback availability is no later than catalogue source close plus the 12h label horizon','inputs_sha256':{'catalogue':sha256(catalogue),'v1':sha256(v1),'v2':sha256(v2),'features':sha256(features)},'outputs_sha256':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()},'counts':{'train_rows':len(train),'test_rows':len(test),'zero_positive_months':monthly.loc[monthly.active_events.eq(0),'month'].tolist()}}
  (stage/'manifest.json').write_text(json.dumps(safe(manifest),indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(f"{sha256(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return manifest
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--catalogue',type=Path,default=CATALOGUE);p.add_argument('--v1',type=Path,default=V1);p.add_argument('--v2',type=Path,default=V2);p.add_argument('--features',type=Path,default=FEATURES);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args(argv);print(json.dumps(safe(run(catalogue=a.catalogue,v1=a.v1,v2=a.v2,features=a.features,output=a.output)),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
