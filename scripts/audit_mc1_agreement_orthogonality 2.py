#!/usr/bin/env python3
"""Causal orthogonality, permutation, lag and June diagnostics for frozen MC1."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_strictfull_prior28_optimizedpolicy_20260812_v1/walkforward_scored_label_ledger.parquet'
MAPS=ROOT/'data_perp/artifacts/strict_r3_recovery_detection_funnel_long_2024may_2026_20260813_v4/multiwindow_ev_maps.parquet'
PRED=ROOT/'data_perp/artifacts/strict_r3_six_mapper_families_long_2025_2026_20260813_v4/finalist_causal_predictions.parquet'
OUT=ROOT/'data_perp/artifacts/strict_r3_mc1_agreement_orthogonality_20260813_v1'
COLS=['candidate_id','__decision_ts__','__symbol__','final_score','base_rank42','conditional_consensus_rank','upstream','ordinary_shadow_consensus_rank','correctness_rank','policy_path_valid','policy_net_bps','policy_label_available_ts']

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);d=pd.read_parquet(LEDGER,columns=COLS);d.__decision_ts__=pd.to_datetime(d.__decision_ts__,utc=True);d.policy_label_available_ts=pd.to_datetime(d.policy_label_available_ts,utc=True);d['day']=d.__decision_ts__.dt.normalize();d['agreement']=d[['base_rank42','conditional_consensus_rank','ordinary_shadow_consensus_rank']].mean(axis=1)
 # Training-only score -> agreement expectation, applied to later held year.
 train=d[(d.__decision_ts__.dt.year.eq(2025))&d.policy_path_valid.fillna(False)&d.policy_net_bps.notna()];held=d[(d.__decision_ts__.dt.year.eq(2026))&d.policy_path_valid.fillna(False)&d.policy_net_bps.notna()].copy();iso=IsotonicRegression(out_of_bounds='clip').fit(train.final_score,train.agreement);held['agreement_orthogonal']=held.agreement-iso.predict(held.final_score)
 held['score_band']=pd.qcut(held.final_score,10,labels=False,duplicates='drop');held['orth_q']=held.groupby('score_band').agreement_orthogonal.transform(lambda x:pd.qcut(x,5,labels=False,duplicates='drop'));surf=held.groupby(['score_band','orth_q']).agg(rows=('candidate_id','size'),ev=('policy_net_bps','mean')).reset_index();audit=surf.groupby('score_band').apply(lambda g:pd.Series({'spearman':g.orth_q.corr(g.ev,method='spearman'),'low_high':g.sort_values('orth_q').ev.iloc[-1]-g.sort_values('orth_q').ev.iloc[0]}),include_groups=False).reset_index();surf.to_parquet(OUT/'orthogonal_agreement_surface.parquet',index=False);audit.to_parquet(OUT/'orthogonal_agreement_audit.parquet',index=False)
 # Null controls: monthly causal fits; permutation is within day x score decile.
 feats=['final_score','agreement','correctness_rank'];rows=[]
 for seed in (17,1729,20260813):
  for control in ('observed','within_day_score_permutation','lagged_day_agreement'):
   pp=[]
   for start in pd.date_range('2026-01-01','2026-07-01',freq='MS',tz='UTC'):
    tr=d[(d.policy_label_available_ts.le(start))&d.__decision_ts__.lt(start)&d.policy_path_valid.fillna(False)&d.policy_net_bps.notna()].copy();va=d[d.__decision_ts__.between(start,start+pd.offsets.MonthBegin(1),inclusive='left')].copy();tr['score_decile']=pd.qcut(tr.final_score,10,labels=False,duplicates='drop')
    if control=='within_day_score_permutation':tr['agreement']=tr.groupby(['day','score_decile']).agreement.transform(lambda x:x.sample(frac=1,random_state=seed).to_numpy())
    if control=='lagged_day_agreement':
     lag=tr.groupby('day').agreement.mean().shift(1);tr['agreement']=tr.day.map(lag);va['agreement']=va.day.map(tr.groupby('day').agreement.mean().shift(1))
    model=HistGradientBoostingRegressor(max_depth=2,max_iter=80,learning_rate=.04,l2_regularization=20,min_samples_leaf=100,random_state=seed).fit(tr[feats].fillna(tr[feats].median()),tr.policy_net_bps);va['mapped']=model.predict(va[feats].fillna(tr[feats].median()));pp.append(va[['candidate_id','mapped']])
   z=d[d.__decision_ts__.dt.year.eq(2026)].merge(pd.concat(pp),on='candidate_id');v=z[z.policy_path_valid.fillna(False)&z.policy_net_bps.notna()&z.mapped.ge(50)];rows.append({'control':control,'seed':seed,'trades':len(v),'ev':v.policy_net_bps.mean(),'total':v.policy_net_bps.sum(),'positive_months':v.assign(month=v.__decision_ts__.dt.strftime('%Y-%m')).groupby('month').policy_net_bps.mean().gt(0).sum()})
 pd.DataFrame(rows).to_parquet(OUT/'null_control_metrics.parquet',index=False)
 # June MC1-only predicted-EV deciles, diagnostic only.
 mc=pd.read_parquet(PRED);mc=mc[mc.arm.eq('MC1_d2')][['candidate_id','mapped_ev']];m21=pd.read_parquet(MAPS,columns=['candidate_id','m21']);j=d[d.__decision_ts__.between('2026-06-01','2026-07-01',inclusive='left')].merge(mc,on='candidate_id').merge(m21,on='candidate_id');j=j[j.policy_path_valid.fillna(False)&j.policy_net_bps.notna()&j.mapped_ev.ge(50)&j.m21.lt(50)].copy();j['decile']=pd.qcut(j.mapped_ev,10,labels=False,duplicates='drop');q=j.groupby('decile').agg(rows=('candidate_id','size'),predicted=('mapped_ev','mean'),realized=('policy_net_bps','mean'),total=('policy_net_bps','sum')).reset_index();q.to_parquet(OUT/'june_mc1_only_predicted_ev_deciles.parquet',index=False)
 (OUT/'manifest.json').write_text(json.dumps({'status':'complete','purpose':'falsification only','training':'strictly prior outcome availability','permutation':'within decision-day x frozen-score decile','promotion_effect':'none'},indent=2)+'\n')
 print('complete')
if __name__=='__main__':main()
