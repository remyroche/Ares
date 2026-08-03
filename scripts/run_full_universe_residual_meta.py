#!/usr/bin/env python3
"""Strict OOF residual-meta window screen for the frozen full-universe base."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
PARAMS=dict(n_estimators=120,learning_rate=.05,num_leaves=24,min_child_samples=300,colsample_bytree=.8,subsample=.8,reg_lambda=10.,random_state=20260801,n_jobs=1,verbosity=-1)
def fam(n):
 for x,g in [('mkt','market'),('xasset','market'),('regime','regime'),('fund','funding'),('oi','open_interest'),('ob_','orderbook'),('vol','volatility'),('ret','returns'),('price','price'),('range','price'),('tail','distribution'),('q_','distribution')]:
  if x in n:return g
 return 'other'
def select(d,cols,y,n=36):
 if len(d)>200_000: ii=np.linspace(0,len(d)-1,200_000,dtype=int); d=d.iloc[ii]; y=y[ii]
 scores=[]
 for c in cols:
  x=pd.to_numeric(d[c],errors='coerce').to_numpy(float); ok=np.isfinite(x)&np.isfinite(y)
  if ok.mean()>=.9 and ok.sum()>500:
   v=spearmanr(x[ok],y[ok]).statistic; scores.append((abs(v) if np.isfinite(v) else -1,c))
 out=[]; counts={}
 for _,c in sorted(scores,reverse=True):
  f=fam(c)
  if c in out or counts.get(f,0)>=5:continue
  out.append(c);counts[f]=counts.get(f,0)+1
  if len(out)==n:break
 if len(out)<30:raise RuntimeError(f'only {len(out)} diverse meta inputs')
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--days',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--geometry',default='tp3_sl2',choices=('tp2_sl1','tp2_sl2','tp3_sl1','tp3_sl2'));p.add_argument('--rolling',action='store_true');p.add_argument('--refit-days',type=int,default=1);p.add_argument('--score-only',action='store_true');p.add_argument('--causal-calibration',action='store_true');p.add_argument('--base-long',type=Path);p.add_argument('--base-short',type=Path);p.add_argument('--meta-train-end');p.add_argument('--eval-start');p.add_argument('--raw-base-output',action='store_true');p.add_argument('--calibrate-raw-meta',action='store_true');p.add_argument('--shared-raw-calibration',action='store_true',help='fit one causal OOF mapping across both sides for a valid pooled book');p.add_argument('--shared-meta-model',action='store_true',help='one direct common-unit meta model with a side indicator; an ablation against side-local meta models');p.add_argument('--side',choices=('long','short'));a=p.parse_args()
 if a.shared_raw_calibration and (not a.raw_base_output or not a.calibrate_raw_meta):
  p.error('--shared-raw-calibration requires --raw-base-output --calibrate-raw-meta')
 if a.shared_meta_model and (a.rolling or not a.raw_base_output or a.side):
  p.error('--shared-meta-model is a static, both-side raw-base-output ablation')
 gross=f't4_{a.geometry}_gross_bps'; net=f't4_{a.geometry}_net_bps'
 au=json.loads(a.audit.read_text()); meta=au['meta']['coverage_ge_90pct']; cols=['candidate_id','__ts__','__label_available_at__','side_name',gross,net]+meta
 raw=pd.concat([pd.read_parquet(x,columns=cols) for x in sorted((a.panel/'parts').glob('*.parquet'))],ignore_index=True);raw.__ts__=pd.to_datetime(raw.__ts__,utc=True);raw.__label_available_at__=pd.to_datetime(raw.__label_available_at__,utc=True)
 def base_prediction_path(side: str, supplied: Path | None) -> Path:
  if supplied:
   return supplied
  # Both layouts are valid checkpoint formats.  The compact side-directory
  # form is what the HPO runner writes; the geometry-prefixed form is the
  # original grid layout.  Prefer an existing file rather than silently
  # falling back to a different base model.
  candidates=(a.base_root/side/'target_screen_predictions.parquet',a.base_root/f't2_{a.geometry}_{side}'/'target_screen_predictions.parquet')
  for candidate in candidates:
   if candidate.exists(): return candidate
  raise FileNotFoundError(f'no base prediction file for {side}; tried {candidates}')
 long_path=base_prediction_path('long',a.base_long); short_path=base_prediction_path('short',a.base_short)
 base_columns=['candidate_id','score_bps'] if not a.raw_base_output else ['candidate_id','p_upper','p_lower','p_timeout']
 pred=pd.concat([pd.read_parquet(long_path),pd.read_parquet(short_path)],ignore_index=True)[base_columns]
 d=raw.merge(pred,on='candidate_id',how='inner',validate='one_to_one'); start=d.__ts__.min().floor('D'); cutoff=start+pd.Timedelta(days=a.days); first_eval=pd.Timestamp(a.eval_start,tz='UTC') if a.eval_start else cutoff+pd.Timedelta(hours=12)
 # Static meta fits use exactly the requested trailing *row-level resolved*
 # window.  Previously --meta-train-end only supplied an upper bound, so a
 # nominal 120-day run silently included all older history.
 meta_end=pd.Timestamp(a.meta_train_end,tz='UTC') if a.meta_train_end else first_eval
 meta_start=meta_end-pd.Timedelta(days=a.days)
 tr=d[d.__label_available_at__.lt(first_eval) & d.__ts__.ge(meta_start) & d.__ts__.lt(meta_end)].copy()
 ev=d[d.__ts__.ge(first_eval)].copy(); outs=[]; contract={}
 if a.rolling:
  # Feature selection is frozen from the first causal window; each following
  # day refits only on the immediately preceding resolved D days.
  selected={}
  for side in ('long','short'):
   x=tr[tr.side_name.eq(side)].copy(); selected[side]=[] if a.score_only else select(x,meta,x[net].to_numpy(float)-x.score_bps.to_numpy(float))
  history={'long':[],'short':[]}
  for day in pd.date_range(first_eval.floor('D'),ev.__ts__.max().floor('D'),freq=f'{a.refit_days}D',tz='UTC'):
   for side in ('long','short'):
    z=ev[(ev.side_name.eq(side))&(ev.__ts__.ge(day))&(ev.__ts__.lt(day+pd.Timedelta(days=a.refit_days)))].copy()
    x=d[(d.side_name.eq(side))&(d.__label_available_at__.lt(day))&(d.__ts__.ge(day-pd.Timedelta(days=a.days+1)))].copy()
    if z.empty or len(x)<500: continue
    chosen=selected[side]; target=x[net].to_numpy(float)-x.score_bps.to_numpy(float)
    xx=x[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32');zz=z[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32');xx=np.column_stack([xx,x.score_bps.to_numpy('float32')]);zz=np.column_stack([zz,z.score_bps.to_numpy('float32')])
    rolling_params={**PARAMS,'n_estimators':40,'learning_rate':.08}
    model=lgb.LGBMRegressor(objective='huber',alpha=.9,**rolling_params).fit(xx,target); bias=float(np.mean(target-model.predict(xx)))
    z=z[['candidate_id','__ts__','__label_available_at__','side_name',gross,net,'score_bps']].copy();z['meta_residual_bps']=model.predict(zz)+bias;z['raw_final_score_bps']=z.score_bps+z.meta_residual_bps
    if a.causal_calibration:
     h=pd.concat(history[side],ignore_index=True) if history[side] else pd.DataFrame()
     h=h[h.__label_available_at__.lt(day)] if len(h) else h
     if len(h)>=500:
      slope, intercept=np.polyfit(h.raw_final_score_bps.to_numpy(float),h[net].to_numpy(float),1); slope=float(np.clip(slope,0.,3.)); z['final_score_bps']=intercept+slope*z.raw_final_score_bps
     else: z['final_score_bps']=z.raw_final_score_bps
    else: z['final_score_bps']=z.raw_final_score_bps
    history[side].append(z.copy());outs.append(z)
  contract={**selected,'mode':'causal rolling fit on individual rows','residual_training_days':a.days,'refit_days':a.refit_days,'score_only':a.score_only,'causal_calibration':a.causal_calibration}
  out=pd.concat(outs,ignore_index=True).sort_values(['final_score_bps','candidate_id'],ascending=[False,True]); rows=[]
  for q in (.01,.05,.1,.2):
   y=out.head(int(len(out)*q+.999));rows.append(dict(top_fraction=q,n=len(y),gross_bps=float(y[gross].mean()),net_bps=float(y[net].mean()),long_n=int(y.side_name.eq('long').sum()),short_n=int(y.side_name.eq('short').sum())))
  a.out.mkdir(parents=True,exist_ok=True);out.to_parquet(a.out/'residual_meta_predictions.parquet',index=False);pd.DataFrame(rows).to_parquet(a.out/'residual_meta_metrics.parquet',index=False);(a.out/'residual_meta_manifest.json').write_text(json.dumps(dict(base=f'strict OOF T2 {a.geometry} tau=.25 predictions',meta_target='realised barrier-exit net bps minus frozen base expected net bps',meta_feature_contract=contract,raw_feature_overlap_with_base='forbidden by audit',meta_train_resolved_before='each evaluation day',eval_start=str(first_eval),days=a.days,mode='rolling'),indent=2));print(pd.DataFrame(rows).to_string(index=False));return
 if a.shared_meta_model:
  # This is intentionally a per-row pooled learner, not a timestamp model or
  # a side quota.  Its target is already in common bps, and side_is_long lets
  # it represent directional asymmetry without separate arbitrary scales.
  target=tr[net].to_numpy(float); chosen=[] if a.score_only else select(tr,meta,target)
  def design(frame):
   context=frame[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32')
   probabilities=frame[['p_upper','p_lower','p_timeout']].to_numpy('float32')
   side=frame.side_name.eq('long').to_numpy('float32')[:,None]
   return np.column_stack([context,probabilities,side])
  xx,zz=design(tr),design(ev); calibration=(1.,0.)
  if a.calibrate_raw_meta:
   split=tr.__ts__.quantile(.5); early=tr[tr.__ts__.lt(split)]; late=tr[tr.__ts__.ge(split)]
   provisional=lgb.LGBMRegressor(objective='huber',alpha=.9,**PARAMS).fit(design(early),early[net].to_numpy(float))
   slope,intercept=np.polyfit(provisional.predict(design(late)),late[net].to_numpy(float),1)
   if not np.isfinite(slope) or slope <= 1e-6: raise RuntimeError(f'degenerate shared-meta OOF calibration slope {slope:.6g}')
   calibration=(float(min(slope,3.)),float(intercept))
  model=lgb.LGBMRegressor(objective='huber',alpha=.9,**PARAMS).fit(xx,target); bias=float(np.mean(target-model.predict(xx)))
  out=ev[['candidate_id','__ts__','side_name',gross,net]].copy(); out['meta_residual_bps']=model.predict(zz)+bias; out['raw_meta_score_bps']=out.meta_residual_bps; out['final_score_bps']=calibration[1]+calibration[0]*out.raw_meta_score_bps
  out=out.sort_values(['final_score_bps','candidate_id'],ascending=[False,True]); rows=[]
  for q in (.01,.05,.1,.2):
   y=out.head(int(len(out)*q+.999)); rows.append(dict(top_fraction=q,n=len(y),gross_bps=float(y[gross].mean()),net_bps=float(y[net].mean()),long_n=int(y.side_name.eq('long').sum()),short_n=int(y.side_name.eq('short').sum())))
  contract={'shared_meta_context_features':chosen,'shared_meta_context_feature_count':len(chosen),'base_inputs':['p_upper','p_lower','p_timeout'],'side_indicator':'side_is_long','raw_feature_overlap_with_base':'forbidden by audit','calibration':{'slope':calibration[0],'intercept':calibration[1],'mode':'pooled chronological OOF' if a.calibrate_raw_meta else 'none'}}
  a.out.mkdir(parents=True,exist_ok=True);out.to_parquet(a.out/'residual_meta_predictions.parquet',index=False);pd.DataFrame(rows).to_parquet(a.out/'residual_meta_metrics.parquet',index=False);(a.out/'residual_meta_manifest.json').write_text(json.dumps(dict(base=f'strict OOF T2 {a.geometry} tau=.25 predictions',meta_mode='shared direct final-score learner over each row’s raw same-side base probabilities',meta_target='realised barrier-exit net bps',base_inputs='same-side p_upper, p_lower, p_timeout plus side_is_long',meta_feature_contract=contract,meta_train_window=[str(meta_start),str(meta_end)],meta_train_resolved_before=str(first_eval),eval_start=str(first_eval),days=a.days),indent=2));print(pd.DataFrame(rows).to_string(index=False));return
 sides=(a.side,) if a.side else ('long','short')
 shared_calibration=None
 if a.shared_raw_calibration:
  # Side models can naturally learn different raw prediction ranges.  The
  # global book needs one common bps mapping, so form strictly chronological
  # provisional predictions for each side, then fit a single map jointly.
  calibration_scores=[]; calibration_actual=[]
  for side in sides:
   x=tr[tr.side_name.eq(side)].copy(); target=x[net].to_numpy(float); chosen=[] if a.score_only else select(x,meta,target)
   split=x.__ts__.quantile(.5); early=x[x.__ts__.lt(split)].copy(); late=x[x.__ts__.ge(split)].copy()
   xe=early[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32'); xl=late[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32')
   xe=np.column_stack([xe,early[['p_upper','p_lower','p_timeout']].to_numpy('float32')]); xl=np.column_stack([xl,late[['p_upper','p_lower','p_timeout']].to_numpy('float32')])
   provisional=lgb.LGBMRegressor(objective='huber',alpha=.9,**PARAMS).fit(xe,early[net].to_numpy(float))
   calibration_scores.append(provisional.predict(xl)); calibration_actual.append(late[net].to_numpy(float))
  slope,intercept=np.polyfit(np.concatenate(calibration_scores),np.concatenate(calibration_actual),1)
  if not np.isfinite(slope) or slope <= 1e-6:
   raise RuntimeError(f'degenerate shared raw-meta OOF calibration slope {slope:.6g}; do not emit a tied global ranking')
  shared_calibration=(float(min(slope,3.)),float(intercept))
 for side in sides:
  x=tr[tr.side_name.eq(side)].copy();z=ev[ev.side_name.eq(side)].copy();target=x[net].to_numpy(float) if a.raw_base_output else x[net].to_numpy(float)-x.score_bps.to_numpy(float); chosen=[] if a.score_only else select(x,meta,target);contract[side]=chosen
  xx=x[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32');zz=z[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32')
  base_input=['p_upper','p_lower','p_timeout'] if a.raw_base_output else ['score_bps'];xx=np.column_stack([xx,x[base_input].to_numpy('float32')]);zz=np.column_stack([zz,z[base_input].to_numpy('float32')])
  calibration=(1.,0.)
  if a.raw_base_output and a.calibrate_raw_meta and not a.shared_raw_calibration:
   split=x.__ts__.quantile(.5); early=x[x.__ts__.lt(split)].copy(); late=x[x.__ts__.ge(split)].copy()
   xe=early[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32');xl=late[chosen].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy('float32');xe=np.column_stack([xe,early[base_input].to_numpy('float32')]);xl=np.column_stack([xl,late[base_input].to_numpy('float32')])
   provisional=lgb.LGBMRegressor(objective='huber',alpha=.9,**PARAMS).fit(xe,early[net].to_numpy(float)); oof=provisional.predict(xl); slope,intercept=np.polyfit(oof,late[net].to_numpy(float),1)
   if not np.isfinite(slope) or slope <= 1e-6:
    raise RuntimeError(f'degenerate {side} raw-meta OOF calibration slope {slope:.6g}; do not emit a tied global ranking')
   calibration=(float(min(slope,3.)),float(intercept));contract[f'{side}_raw_meta_calibration']={'slope':calibration[0],'intercept':calibration[1],'oof_window_start':str(split)}
  if shared_calibration is not None:
   calibration=shared_calibration
   contract['shared_raw_meta_calibration']={'slope':calibration[0],'intercept':calibration[1],'fit':'pooled side-local provisional predictions on later resolved meta-training rows'}
  model=lgb.LGBMRegressor(objective='huber',alpha=.9,**PARAMS).fit(xx,target)
  # A side-local intercept is a causal common-unit mapping, estimated from
  # resolved meta-training rows only.  It neither changes within-side ranks
  # nor creates a quota; it removes a systematic level bias before the one
  # pooled global ranking.
  residual_bias=float(np.mean(target-model.predict(xx)))
  z=z[['candidate_id','__ts__','side_name',gross,net]].copy();z['meta_residual_bps']=model.predict(zz)+residual_bias
  # In the residual architecture, meta predicts the correction and must be
  # added back to the frozen base common-bps score.  Raw-base mode instead is
  # a direct final-score learner over the three same-side probabilities.
  z['raw_meta_score_bps']=z.meta_residual_bps if a.raw_base_output else z.meta_residual_bps + ev.loc[z.index, 'score_bps'].to_numpy(float)
  z['final_score_bps']=calibration[1]+calibration[0]*z.raw_meta_score_bps if a.raw_base_output else z.raw_meta_score_bps
  contract[f'{side}_residual_bias_bps']=residual_bias
  contract[f'{side}_base_inputs']=base_input
  contract[f'{side}_meta_context_feature_count']=len(chosen)
  outs.append(z)
 out=pd.concat(outs,ignore_index=True).sort_values(['final_score_bps','candidate_id'],ascending=[False,True]);rows=[]
 for q in (.01,.05,.1,.2):
  y=out.head(int(len(out)*q+.999));rows.append(dict(top_fraction=q,n=len(y),gross_bps=float(y[gross].mean()),net_bps=float(y[net].mean()),long_n=int(y.side_name.eq('long').sum()),short_n=int(y.side_name.eq('short').sum())))
 a.out.mkdir(parents=True,exist_ok=True);out.to_parquet(a.out/'residual_meta_predictions.parquet',index=False);pd.DataFrame(rows).to_parquet(a.out/'residual_meta_metrics.parquet',index=False);(a.out/'residual_meta_manifest.json').write_text(json.dumps(dict(base=f'strict OOF T2 {a.geometry} tau=.25 predictions',meta_mode='direct final-score learner over same-side raw base probabilities' if a.raw_base_output else 'residual correction added to frozen base score',meta_target='realised barrier-exit net bps' if a.raw_base_output else 'realised barrier-exit net bps minus frozen base expected net bps',base_inputs='same-side p_upper, p_lower, p_timeout' if a.raw_base_output else 'same-side frozen base common-bps score',global_score_calibration='shared pooled-side causal OOF affine mapping' if a.shared_raw_calibration else ('side-local causal OOF affine mapping' if a.calibrate_raw_meta else 'none'),meta_feature_contract=contract,raw_feature_overlap_with_base='forbidden by audit',meta_train_resolved_before=str(first_eval),eval_start=str(first_eval),days=a.days),indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
