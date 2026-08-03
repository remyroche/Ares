#!/usr/bin/env python3
"""Pre-2026, timestamp-level residual-book risk calibration diagnostic.

There is exactly one statistical training row per UTC hour and arm.  Candidate
rows are used only to materialise a residual-selected historical book and to
broadcast held-hour predictions back into an explicitly diagnostic policy
replay.  No 2026 file is read.
"""
from __future__ import annotations
import hashlib, json, math, os, shutil, tempfile, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data_perp/artifacts'
SRC=ART/'pre2026_oof_model_failure_incremental_value_20260730_v3'
OUT=ART/'pre2026_hourly_book_risk_calibrator_20260730_v2_r1'
CORE=['base_score','residual_score','residual_minus_base']
REGIME=['regime_change_probability_mean','regime_change_probability_max','regime_run_length_mean','regime_run_length_q05','regime_run_length_entropy','regime_signal_count','regime_state_age_hours','regime_is_persistent_24h','regime_is_persistent_72h']
TRANSITION=['transition_lgbm_probability','transition_lgbm_entropy','transition_lgbm_margin','transition_bocpd_stable_probability','transition_bocpd_onset_h1_probability','transition_bocpd_onset_h3_probability','transition_bocpd_onset_h6_probability','transition_bocpd_onset_h12_probability']
TRAJECTORY=['trajectory_available','trajectory_transition_probability','trajectory_probability_entropy','trajectory_top2_margin']
ARMS={'regime':REGIME,'transition':TRANSITION,'trajectory':TRAJECTORY,'combined':REGIME+TRANSITION+TRAJECTORY}
# The hurdle is fit on every arm-available hour.  All remaining labels are
# conditional on opportunity; this prevents zero-selected hours being silently
# dropped then promoted by the broadcast.
TARGETS={'book_opportunity':False,'book_selected_count_if_opportunity':True,
         'book_mean_net_ev_if_selected':True,'book_sum_net_ev_if_selected':True,
         'book_failure_rate_if_selected':True,'book_downside_severity_if_selected':True}
GAMMAS=[0.0,0.05,0.10,0.25]; TOP=.10

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def digest(s): return hashlib.sha256('|'.join(pd.Series(s).astype(str).sort_values()).encode()).hexdigest()
def sealed(p): return sha(p/'manifest.json')==(p/'manifest.sha256').read_text().split()[0]

def weighted_corr(x,y,w):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float); ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
 if ok.sum()<3:return np.nan
 x=x[ok];y=y[ok];w=w[ok]; mx=np.average(x,weights=w);my=np.average(y,weights=w)
 vx=np.average((x-mx)**2,weights=w);vy=np.average((y-my)**2,weights=w)
 return np.average((x-mx)*(y-my),weights=w)/math.sqrt(vx*vy) if vx>0 and vy>0 else np.nan
def weighted_spearman(x,y,w): return weighted_corr(pd.Series(x).rank().to_numpy(),pd.Series(y).rank().to_numpy(),w)
def model(): return Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=30.0))])
def opportunity_model(): return Pipeline([('scale',StandardScaler()),('logit',LogisticRegression(C=.02,max_iter=300,class_weight='balanced',random_state=17))])
def availability(x,arm):
 f=CORE+ARMS[arm]; ok=x[f].notna().all(axis=1)
 if arm in ('regime','combined'):ok&=x.bocpd_regime_available.fillna(False)
 if arm in ('transition','combined'):ok&=x.lgbm_transition_available.fillna(False)
 if arm in ('trajectory','combined'):ok&=x.trajectory_available.fillna(False)
 return ok

def hourly_book(x,arm):
 """One row for every arm-available hour; selected_count is a weight only."""
 f=CORE+ARMS[arm]; z=x.loc[availability(x,arm)].copy(); sel=z[z.residual_selected_global_top10].copy()
 # inference-time CORE summaries are computed from all arm-available candidates.
 rows=[]; selected_by_hour={ts:g for ts,g in sel.groupby('__ts__',sort=False)}
 for ts,g in z.groupby('__ts__',sort=True):
  s=selected_by_hour.get(ts)
  if s is None: s=sel.iloc[:0]
  r={'__ts__':ts,'era':g.era.iloc[0],'selected_count':len(s),'candidate_count':len(g),
     'book_opportunity':int(len(s)>0),'book_selected_count_if_opportunity':float(len(s)) if len(s) else np.nan,
     'book_mean_net_ev_if_selected':s.execution_net_ev_12h.mean() if len(s) else np.nan,
     'book_sum_net_ev_if_selected':s.execution_net_ev_12h.sum() if len(s) else np.nan,
     'book_failure_rate_if_selected':s.execution_net_ev_12h.le(0).mean() if len(s) else np.nan,
     'book_downside_severity_if_selected':(-s.execution_net_ev_12h.clip(upper=0)).mean() if len(s) else np.nan,
     'selected_long_fraction':s.side_name.eq('long').mean() if len(s) else np.nan,'selected_long_count':s.side_name.eq('long').sum(),'selected_short_count':s.side_name.eq('short').sum()}
  for c in CORE:
   a=g[c].to_numpy(float); k=min(5,len(a)); order=np.sort(a)
   r.update({f'{c}__max':float(np.max(a)),f'{c}__q90':float(np.quantile(a,.9)),f'{c}__top5mean':float(order[-k:].mean()),f'{c}__std':float(np.std(a))})
  # Context is hourly by construction; validate it is not candidate-specific.
  for c in ARMS[arm]:
   if g[c].nunique(dropna=False)>1: raise RuntimeError(f'candidate-varying hourly context: {arm}/{c}/{ts}')
   r[c]=g[c].iloc[0]
  rows.append(r)
 return pd.DataFrame(rows)

def hourly_features(arm,context):
 base=['candidate_count']+[f'{c}__{s}' for c in CORE for s in ['max','q90','top5mean','std']]
 return base+(ARMS[arm] if context else [])

def hour_metrics(p):
 out=[]
 for (arm,kind,target,era),g in p.groupby(['arm','kind','target','era'],sort=True):
  g=g.dropna(subset=['actual'])
  if g.empty:continue
  w=np.where(g.target.eq('book_opportunity'),1.0,g.selected_count).astype(float); a=g.actual.to_numpy(float);q=g.prediction.to_numpy(float)
  row={'arm':arm,'kind':kind,'target':target,'era':era,'hours':len(g),'selected_weight':w.sum(),
       'weighted_rank_ic':weighted_spearman(q,a,w),'weighted_mae':np.average(np.abs(q-a),weights=w),
       'weighted_bias':np.average(q-a,weights=w)}
  if target in ('book_opportunity','book_failure_rate_if_selected'):row['weighted_brier']=np.average((q-a)**2,weights=w)
  if target=='book_opportunity':
   row.update({'opportunity_auc':roc_auc_score(a,q) if pd.Series(a).nunique()==2 else np.nan,
               'opportunity_actual_rate':a.mean(),'opportunity_prediction_mean':q.mean()})
  out.append(row)
 return pd.DataFrame(out)

def select_top(g,score):
 n=math.ceil(len(g)*TOP); return g.assign(_score=score).sort_values(['_score','candidate_id'],ascending=[False,True],kind='stable').index[:n]
def policy_metrics(frame,arm,gamma):
 rows=[]
 for era,g in frame.groupby('era',sort=True):
  differential=0.0 if gamma==0 else (g.context_hour_expected_contribution-g.score_only_hour_expected_contribution)
  corrected=g.residual_score+gamma*differential
  idx=select_top(g,corrected); raw=select_top(g,g.residual_score);chosen=g.loc[idx].copy();chosen['corrected_score']=corrected.loc[idx]
  ev=chosen.execution_net_ev_12h
  week=chosen.assign(week=chosen.__ts__.dt.to_period('W').astype(str)).groupby('week').execution_net_ev_12h.mean()
  month=chosen.assign(month=chosen.__ts__.dt.to_period('M').astype(str)).groupby('month').execution_net_ev_12h.mean()
  share=chosen.__symbol__.value_counts(normalize=True)
  rows.append({'arm':arm,'gamma':gamma,'era':era,'held_era_end_utc':chosen.__ts__.max(),'selected_rows':len(chosen),'net_ev':ev.mean(),'rank_ic':chosen.corrected_score.corr(ev,method='spearman'),
   'long_net_ev':chosen.loc[chosen.side_name.eq('long'),'execution_net_ev_12h'].mean(),'short_net_ev':chosen.loc[chosen.side_name.eq('short'),'execution_net_ev_12h'].mean(),
   'weekly_q10':week.quantile(.1),'weekly_q50':week.quantile(.5),'monthly_q10':month.quantile(.1),'monthly_q50':month.quantile(.5),
   'turnover_vs_residual_control':1-len(set(idx)&set(raw))/len(idx),'asset_hhi':float((share**2).sum())})
 return pd.DataFrame(rows)

def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 if not sealed(SRC):raise RuntimeError('unsealed v3 source')
 common=['candidate_id','__ts__','__symbol__','side_name','era','execution_label_end_utc','execution_net_ev_12h','base_score','residual_score','residual_minus_base','residual_selected_global_top10','bocpd_regime_available','lgbm_transition_available','trajectory_available']
 x=pd.read_parquet(SRC/'materialized_targets.parquet',columns=list(dict.fromkeys(common+sum(ARMS.values(),[]))))
 x.__ts__=pd.to_datetime(x.__ts__,utc=True);x.execution_label_end_utc=pd.to_datetime(x.execution_label_end_utc,utc=True)
 if x.candidate_id.duplicated().any() or x.__ts__.dt.minute.ne(0).any() or x.__ts__.dt.second.ne(0).any() or x.execution_label_end_utc.le(x.__ts__).any() or x.execution_label_end_utc.ge(pd.Timestamp('2026-01-01',tz='UTC')).any():raise RuntimeError('fail closed cadence/identity/label boundary')
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'));pred=[];audit=[];hour_frames={};candidate_scores=[];support=[];integrity=[]
 try:
  for arm in ARMS:
   h=hourly_book(x,arm)
   if h.__ts__.duplicated().any() or h.__ts__.dt.minute.ne(0).any():raise RuntimeError('fail closed: hourly materialization not unique/aligned')
   hour_frames[arm]=h; h.to_parquet(stage/f'hourly_book_{arm}.parquet',index=False)
   for era,te_all in h.groupby('era',sort=True):
    tr_all=h[h.era.ne(era)]
    for kind,ctx in [('score_only',False),('context',True)]:
     fs=hourly_features(arm,ctx)
     for target,conditional in TARGETS.items():
      tr=tr_all.dropna(subset=[target]) if conditional else tr_all
      te=te_all # Conditional heads intentionally predict zero-opportunity hours too.
      if target=='book_opportunity':
       m=opportunity_model().fit(tr[fs],tr[target]);q=m.predict_proba(te[fs])[:,1]
      else:
       m=model().fit(tr[fs],tr[target],ridge__sample_weight=tr.selected_count.to_numpy(float));q=m.predict(te[fs])
      pred.append(pd.DataFrame({'arm':arm,'kind':kind,'target':target,'era':era,'__ts__':te.__ts__.to_numpy(),'selected_count':te.selected_count.to_numpy(),'actual':te[target].to_numpy(),'prediction':q}))
      audit.append({'arm':arm,'kind':kind,'target':target,'held_era':era,'train_hours':len(tr),'test_hours':len(te),'train_hour_sha256':digest(tr.__ts__),'test_hour_sha256':digest(te.__ts__),'feature_count':len(fs),'features':'|'.join(fs),'selected_weight_train':tr.selected_count.sum(),'selected_weight_test':te.selected_count.sum(),'unit_weight_training':target=='book_opportunity','predicts_all_held_hours':True})
  p=pd.concat(pred,ignore_index=True);p.to_parquet(stage/'hourly_oof_predictions.parquet',index=False);met=hour_metrics(p);met.to_csv(stage/'hourly_fold_metrics.csv',index=False);pd.DataFrame(audit).to_csv(stage/'hourly_fold_audit.csv',index=False)
  for arm,h in hour_frames.items():
   for target in TARGETS:
    for kind in ['score_only','context']:
     actual=p[(p.arm.eq(arm))&(p.target.eq(target))&(p.kind.eq(kind))]
     integrity.append({'check':'hourly_prediction_coverage','arm':arm,'target':f'{kind}:{target}','expected_rows':len(h),'actual_rows':len(actual),'exact':len(actual)==len(h)})
  opp=p[p.target.eq('book_opportunity')].copy();opp['calibration_decile']=opp.groupby(['arm','kind','era']).prediction.transform(lambda s:pd.qcut(s.rank(method='first'),10,labels=False,duplicates='drop'))
  opp.groupby(['arm','kind','era','calibration_decile'],as_index=False).agg(hours=('actual','size'),actual_opportunity_rate=('actual','mean'),predicted_opportunity=('prediction','mean')).to_csv(stage/'all_hour_opportunity_calibration.csv',index=False)
  # Expected hourly contribution = hurdle probability × conditional selected
  # count × conditional mean value. It is zero-adjusted for unsupported hours.
  for arm,h in hour_frames.items():
   pieces=[]
   for target in ['book_opportunity','book_selected_count_if_opportunity','book_mean_net_ev_if_selected']:
    z=p[(p.arm.eq(arm))&(p.target.eq(target))].pivot(index=['era','__ts__'],columns='kind',values='prediction').reset_index()
    if set(z.columns)<{'score_only','context'}:raise RuntimeError('missing prediction pair')
    pieces.append(z.rename(columns={'score_only':f'score_only__{target}','context':f'context__{target}'}))
   a=pieces[0].merge(pieces[1],on=['era','__ts__'],validate='one_to_one').merge(pieces[2],on=['era','__ts__'],validate='one_to_one')
   for kind in ['score_only','context']:
    a[f'{kind}_hour_expected_contribution']=(a[f'{kind}__book_opportunity'].clip(0,1)*a[f'{kind}__book_selected_count_if_opportunity'].clip(lower=0)*a[f'{kind}__book_mean_net_ev_if_selected'])
   # Preserve every candidate in the global book. An hour unavailable to this
   # arm receives exactly zero differential correction.
   c=x[['candidate_id','__ts__','__symbol__','side_name','era','residual_score','execution_net_ev_12h']].merge(a[['era','__ts__','score_only_hour_expected_contribution','context_hour_expected_contribution']],on=['era','__ts__'],how='left',validate='many_to_one')
   c['hour_context_supported']=c.context_hour_expected_contribution.notna()
   c[['score_only_hour_expected_contribution','context_hour_expected_contribution']]=c[['score_only_hour_expected_contribution','context_hour_expected_contribution']].fillna(0.0)
   candidate_scores.append(c.assign(arm=arm))
   hs=h.groupby('era',as_index=False).agg(hour_rows=('__ts__','size'),zero_opportunity_hours=('book_opportunity',lambda s:int((s==0).sum())),opportunity_rate=('book_opportunity','mean'))
   cs=c.groupby('era',as_index=False).agg(candidate_rows=('candidate_id','size'),context_supported_candidate_rows=('hour_context_supported','sum'))
   support.append(hs.merge(cs,on='era',how='outer').assign(arm=arm,fully_supported_for_gate=lambda d:d.hour_rows.notna() & d.context_supported_candidate_rows.gt(0)))
  candidate=pd.concat(candidate_scores,ignore_index=True);candidate.to_parquet(stage/'candidate_oof_broadcast_scores.parquet',index=False)
  support=pd.concat(support,ignore_index=True);support.to_csv(stage/'arm_era_hourly_and_candidate_availability.csv',index=False)
  econ=pd.concat([policy_metrics(candidate[candidate.arm.eq(arm)],arm,g) for arm in ARMS for g in GAMMAS],ignore_index=True);econ.to_csv(stage/'context_incremental_policy_economics.csv',index=False)
  summaries=[]
  for (arm,gamma),z in econ.groupby(['arm','gamma'],sort=True):
   latest=z.loc[z.held_era_end_utc.idxmax()]
   w=z.selected_rows.to_numpy(float)
   summaries.append({'arm':arm,'gamma':gamma,'aggregate_selected_rows':int(z.selected_rows.sum()),
    'aggregate_net_ev':float(np.average(z.net_ev,weights=w)),'aggregate_rank_ic':float(np.average(z.rank_ic,weights=w)),
    'aggregate_long_net_ev':float(np.average(z.long_net_ev,weights=w)),'aggregate_short_net_ev':float(np.average(z.short_net_ev,weights=w)),
    'aggregate_weekly_q10_mean':float(np.average(z.weekly_q10,weights=w)),'aggregate_weekly_q50_mean':float(np.average(z.weekly_q50,weights=w)),
    'aggregate_monthly_q10_mean':float(np.average(z.monthly_q10,weights=w)),'aggregate_monthly_q50_mean':float(np.average(z.monthly_q50,weights=w)),
    'aggregate_turnover_mean':float(np.average(z.turnover_vs_residual_control,weights=w)),'aggregate_asset_hhi_mean':float(np.average(z.asset_hhi,weights=w)),
    'latest_era':latest.era,'latest_net_ev':latest.net_ev,'latest_rank_ic':latest.rank_ic,'latest_long_net_ev':latest.long_net_ev,'latest_short_net_ev':latest.short_net_ev,
    'latest_weekly_q10':latest.weekly_q10,'latest_weekly_q50':latest.weekly_q50,'latest_monthly_q10':latest.monthly_q10,'latest_monthly_q50':latest.monthly_q50,
    'latest_turnover':latest.turnover_vs_residual_control,'latest_asset_hhi':latest.asset_hhi})
  pd.DataFrame(summaries).to_csv(stage/'context_incremental_policy_summary.csv',index=False)
  deltas=[];gates=[]
  for arm in ARMS:
   control=econ[(econ.arm.eq(arm))&(econ.gamma.eq(0.0))].sort_values('era'); stable=bool(control.net_ev.median()>0 and (control.net_ev>0).mean()>=.75 and control.net_ev.min()>=-.01)
   for gamma in GAMMAS[1:]:
    q=econ[(econ.arm.eq(arm))&(econ.gamma.eq(gamma))].merge(control[['era','net_ev']],on='era',suffixes=('','_control'))
    base=control[['era','weekly_q10','weekly_q50','monthly_q10','monthly_q50','long_net_ev','short_net_ev','turnover_vs_residual_control','asset_hhi']]
    q=q.merge(base,on='era',suffixes=('','_control'))
    for col in ['weekly_q10','weekly_q50','monthly_q10','monthly_q50','long_net_ev','short_net_ev','turnover_vs_residual_control','asset_hhi']:
     q[f'{col}_delta']=q[col]-q[f'{col}_control']
    q=q.merge(support[support.arm.eq(arm)][['era','fully_supported_for_gate']],on='era',how='left').assign(arm=arm,gamma=gamma,net_ev_delta=lambda d:d.net_ev-d.net_ev_control)
    deltas.append(q)
    e=q[q.fully_supported_for_gate.fillna(False)].copy();d=e.net_ev_delta
    economic=bool(len(e)>=6 and d.median()>0 and (d>0).mean()>=.75 and d.min()>=-.002)
    tails_sides=bool(len(e)>=6 and e.weekly_q10_delta.median()>=0 and e.weekly_q50_delta.median()>=0 and e.monthly_q10_delta.median()>=0 and e.monthly_q50_delta.median()>=0 and e.long_net_ev_delta.median()>=0 and e.short_net_ev_delta.median()>=0)
    gates.append({'arm':arm,'gamma':gamma,'matched_eras':len(e),'median_net_ev_delta':d.median(),'min_net_ev_delta':d.min(),'positive_era_fraction':(d>0).mean(),'median_weekly_q10_delta':e.weekly_q10_delta.median(),'median_weekly_q50_delta':e.weekly_q50_delta.median(),'median_monthly_q10_delta':e.monthly_q10_delta.median(),'median_monthly_q50_delta':e.monthly_q50_delta.median(),'median_long_net_ev_delta':e.long_net_ev_delta.median(),'median_short_net_ev_delta':e.short_net_ev_delta.median(),'residual_control_stable':stable,'context_economics_incremental':economic,'tails_and_both_sides_guard':tails_sides,'eligible':bool(stable and economic and tails_sides)})
  pd.concat(deltas,ignore_index=True).to_csv(stage/'context_incremental_policy_deltas.csv',index=False)
  gate=pd.DataFrame(gates);gate.to_csv(stage/'context_incremental_economics_gate.csv',index=False)
  integrity.append({'check':'base_ledger_1h_and_pre2026_labels','arm':'all','target':'all','expected_rows':len(x),'actual_rows':len(x),'exact':True})
  pd.DataFrame(integrity).to_csv(stage/'hourly_design_integrity_audit.csv',index=False)
  contract={'schema':'pre2026_hourly_book_risk_calibrator_v2','status':'SEALED_PRE2026_TIMESTAMP_LEVEL_DIAGNOSTIC_NON_PROMOTION','promotion_eligible':False,'scope':'sealed v3 pre-2026 materialized ledger only; no 2026 input','decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','unit_of_observation':'one statistical row for every arm-available UTC hour; selected_count is a sample weight only for conditional heads, never candidate replication','targets':{'book_opportunity':'all-hour selected residual-global-top10 count >0','book_selected_count_if_opportunity':'selected count, conditional on opportunity','book_mean_net_ev_if_selected':'mean selected residual-global-top10 net EV, conditional on opportunity','book_sum_net_ev_if_selected':'sum selected residual-global-top10 net EV, conditional on opportunity','book_failure_rate_if_selected':'selected net EV <=0 fraction, conditional on opportunity','book_downside_severity_if_selected':'mean max(0,-net EV), conditional on opportunity'},'inference_inputs':'hourly candidate count and CORE max/q90/top5mean/std plus arm-local hourly regime/transition/trajectory fields','learner':'hourly pooled fixed StandardScaler+LogisticRegression(C=.02,class_weight=balanced) for all-hour opportunity; fixed StandardScaler+Ridge(alpha=30) with selected-count sample weights for conditional heads; leave-era-out; no HPO','policy_correction':'residual_score + gamma*(context expected hourly contribution - arm-matched score-only expected hourly contribution), where contribution=clip(opportunity,0,1)*clip(conditional_count,0,+inf)*conditional_mean_EV; conditional heads predict every held hour and arm-unsupported hours are zero-adjusted','gamma_grid':GAMMAS,'selection':'one pooled global top10 per fixed held era, ties corrected_score desc/candidate_id asc','matched_evidence':'fully unsupported arm/eras are retained as zero-adjusted policy candidates but excluded from context-vs-score gate evidence','gate':'context economic delta median>0, >=75% positive eras, min>=-.002, >=6 matched supported eras; weekly/monthly Q10/Q50 and long/short median deltas >=0; AND residual control median EV>0, >=75% positive, min>=-.01','environment':{'python':sys.version,'pandas':pd.__version__,'sklearn':sklearn.__version__},'implementation_sha256':{str(Path(__file__).resolve()):sha(Path(__file__))}}
  dump(stage/'contract.json',contract);files=[f for f in stage.iterdir() if f.is_file()];man={'schema':contract['schema'],'status':contract['status'],'promotion_eligible':False,'contract':contract,'inputs_sha256':{str((SRC/'manifest.json').resolve()):sha(SRC/'manifest.json'),str((SRC/'materialized_targets.parquet').resolve()):sha(SRC/'materialized_targets.parquet'),str(Path(__file__).resolve()):sha(Path(__file__))},'outputs_sha256':{f.name:sha(f) for f in files}};dump(stage/'manifest.json',man);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
