#!/usr/bin/env python3
"""D1 only: chronological cross-fitted future-teacher distillation for T2."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import BarrierGeometry,soft_event_targets,top_book_metrics
from scripts.run_t2_atr_sequential_funnel import _add_causal_context,_fit_base,_huber,_score_frame

FUTURE=("__peak_mfe_atr_12h__","__mae_before_meaningful_mfe_atr_12h__","__time_to_first_meaningful_mfe_hours_12h__","__future_slope_atr_per_hour_12h__","__mfe_mae_path_efficiency_12h__","__mfe_integral_path_efficiency_12h__","__mfe_timing_path_efficiency_12h__","__mfe_persistence_path_efficiency_12h__","__peak_mfe_fraction_above_80pct_12h__","__mfe_ratio_to_peak_at_8h_12h__","__adverse_trough_atr_12h__","__adverse_trough_recovery_fraction_12h__","clean_economic_favorable_first","adverse_first","first_favorable_minute","first_adverse_minute","first_event_minute","endpoint_signed_return")
def add_labels(base,event):
 d=base.merge(event,on='candidate_id',how='left',validate='one_to_one')
 if d.timeout.isna().any():raise ValueError('canonical event label gap')
 d[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']]=soft_event_targets(d,BarrierGeometry(2,1),temperature_atr=.25);return d
def score(frame,values,variant):
 out=top_book_metrics(_score_frame(frame,values,variant,'distillation_development','TP2_SL1',.25),score_column='score_bps');out['variant']=variant;return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--features-json',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 fs=list(validate_feature_columns(json.loads(a.features_json.read_text())['raw_feature_columns']))
 req={'candidate_id','side_name','__ts__','__symbol__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','oof_fold'}
 led=_add_causal_context(pd.read_parquet(a.ledger,columns=sorted(req|set(fs))))
 event=pd.read_parquet(a.events)
 support=pd.read_parquet(a.support,columns=['candidate_id',*FUTURE])
 data=add_labels(led,event).merge(support,on='candidate_id',validate='one_to_one')
 data['__ts__']=pd.to_datetime(data['__ts__'],utc=True)
 train=data[data.oof_fold.eq('base_train')].sort_values('__ts__').reset_index(drop=True);dev=data[data.oof_fold.eq('meta_train')].copy()
 future=train.loc[:,FUTURE].to_numpy(np.float32);real=train.loc[:,['t2_upper_soft','t2_lower_soft','t2_timeout_soft']].to_numpy(float)
 teacher=np.full_like(real,np.nan);cuts=np.array_split(np.arange(len(train)),3)
 lineage=[]
 for pos in (1,2):
  fit=np.concatenate(cuts[:pos]);test=cuts[pos]
  pred=np.column_stack([np.maximum(_huber(future[fit],real[fit,j],future[test]),0) for j in range(3)])
  teacher[test]=pred/np.maximum(pred.sum(1,keepdims=True),1e-8)
  lineage.append({'inner_fold':pos,'fit_rows':len(fit),'test_rows':len(test),'fit_end':str(train.__ts__.iloc[fit[-1]]),'teacher_cross_fitted':True})
 valid=np.isfinite(teacher).all(1);blended=real.copy();blended[valid]=.5*real[valid]+.5*teacher[valid]
 distilled=train.copy();distilled[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']]=blended
 base_score,base_prob=_fit_base(train,dev,fs);distill_score,distill_prob=_fit_base(distilled,dev,fs)
 stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  pd.DataFrame({'candidate_id':train.candidate_id,'teacher_p_upper':teacher[:,0],'teacher_p_lower':teacher[:,1],'teacher_p_timeout':teacher[:,2],'teacher_cross_fitted':valid}).to_parquet(stage/'teacher_oof_predictions.parquet',index=False)
  pd.concat([score(dev,base_score,'D0_no_distillation'),score(dev,distill_score,'D1_base_distillation')]).to_parquet(stage/'distillation_ablation.parquet',index=False)
  (stage/'future_teacher_manifest.json').write_text(json.dumps({'target':'canonical T2 TP2_SL1 soft probabilities','teacher_future_features':list(FUTURE),'alpha':.5,'temperature':1,'teacher_crossfit':'three chronological blocks; first warm-up block receives no teacher target','teacher_outputs_never_in_inference':True,'lineage':lineage},indent=2)+'\n')
  os.replace(stage,a.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
