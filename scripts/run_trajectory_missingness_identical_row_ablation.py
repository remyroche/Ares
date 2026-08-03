#!/usr/bin/env python3
"""Execute sealed five-arm trajectory missingness identical-row ablation."""
from __future__ import annotations
import hashlib,json,math,os,shutil,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';PRE=ART/'trajectory_missingness_identical_row_ablation_preregistration_20260730_v1';V=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3';S=ART/'authoritative_soft_regime_transition_sidecars_20260730_v1';T=ART/'hourly_trajectory_transition_soft_sidecar_20260730_v1';OUT=ART/'trajectory_missingness_identical_row_ablation_20260730_v1';TOP=.10
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def features(name):
 r=['raw_score'];g=['regime_change_probability_mean','regime_state_age_hours'];e=['transition_lgbm_probability','transition_lgbm_entropy','transition_lgbm_margin'];t=['trajectory_available','trajectory_transition_probability','probability_entropy','top2_margin']
 return r+(e if name in ['baseline_existing_transition_control','existing_transition_plus_trajectory','regime_plus_existing_transition_plus_trajectory'] else [])+(t if 'trajectory' in name else [])+(g if name.startswith('regime_') else [])
def fit_predict(tr,te,fs):
 x=tr[fs].apply(pd.to_numeric,errors='coerce');z=te[fs].apply(pd.to_numeric,errors='coerce');med=x.median().fillna(0);m=Pipeline([('s',SplineTransformer(n_knots=4,degree=2,knots='quantile',include_bias=False)),('r',Ridge(alpha=8.0))]).fit(x.fillna(med),tr.execution_net_ev_12h);return m.predict(z.fillna(med))
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 for r in [PRE,V,S,T]:
  if (r/'manifest.sha256').read_text().split()[0]!=sha(r/'manifest.json'):raise RuntimeError('unsealed')
 vm=json.loads((V/'manifest.json').read_text());hp=V/'historical_oof_scores.parquet';fp=V/'frozen_2026_candidate_scores.parquet'
 h=pd.read_parquet(hp,filters=[('arm','==','baseline')]);f=pd.read_parquet(fp,filters=[('arm','==','baseline')]);h['raw_score']=h.raw_score;f['raw_score']=f.score_residual_expected_ev
 for x in [h,f]:x['__ts__']=pd.to_datetime(x.__ts__,utc=True)
 rm=pd.read_parquet(S/'soft_regime_hourly.parquet');tm=pd.read_parquet(S/'soft_transition_hourly.parquet');ctx=rm.merge(tm,on='source_utc',validate='one_to_one',suffixes=('','_t')).rename(columns={'bocpd__change_probability_mean':'regime_change_probability_mean','bocpd__state_age_hours':'regime_state_age_hours','lgbm_transition_probability':'transition_lgbm_probability','lgbm_entropy':'transition_lgbm_entropy','lgbm_margin':'transition_lgbm_margin'})
 q=pd.read_parquet(T/'hourly_trajectory_transition_soft_sidecar.parquet');q.source_utc=pd.to_datetime(q.source_utc,utc=True)
 def join(x):
  x=x.drop(columns=[c for c in ctx.columns if c!='source_utc' and c in x.columns],errors='ignore').merge(ctx,left_on='__ts__',right_on='source_utc',how='left',validate='many_to_one').drop(columns='source_utc');x=x.merge(q[['source_utc','trajectory_available','trajectory_transition_probability','probability_entropy','top2_margin']],left_on='__ts__',right_on='source_utc',how='left',validate='many_to_one').drop(columns='source_utc');x['trajectory_available']=x.trajectory_available.fillna(False).astype(float);x['trajectory_transition_probability']=x.trajectory_transition_probability.fillna(.5);x['probability_entropy']=x.probability_entropy.fillna(np.log(2));x['top2_margin']=x.top2_margin.fillna(0.);return x
 h,f=join(h),join(f)
 arms=['baseline_existing_transition_control','trajectory_availability_neutral_only','existing_transition_plus_trajectory','regime_plus_trajectory','regime_plus_existing_transition_plus_trajectory'];rows=[];per=[];side=[];avail=[];selected=[]
 for n,a in enumerate(arms):
  fs=features(a);o=[]
  for block in pd.date_range(pd.Timestamp('2023-01-01',tz='UTC'),h.__ts__.max().normalize(),freq='3MS',tz='UTC'):
   te=h[(h.__ts__>=block)&(h.__ts__<block+pd.DateOffset(months=3))];tr=h[h.__ts__<block]
   if len(te) and len(tr)>1000:
    for s,z in te.groupby('side_name'):o.append(z.assign(raw_oof=fit_predict(tr[tr.side_name.eq(s)],z,fs)))
  oo=pd.concat(o);from sklearn.isotonic import IsotonicRegression
  iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(oo.raw_oof,oo.execution_net_ev_12h);parts=[]
  for s,z in f.groupby('side_name'):parts.append(z.assign(raw_score=fit_predict(h[h.side_name.eq(s)],z,fs)))
  z=pd.concat(parts);z['mapped_score']=iso.predict(z.raw_score);z=z.sort_values(['mapped_score','raw_score','candidate_id'],ascending=[False,False,True],kind='stable');z['selected_global_top10']=False;z.loc[z.index[:math.ceil(len(z)*TOP)],'selected_global_top10']=True;p=z[z.selected_global_top10].copy();rows.append({'arm':a,'candidate_rows':len(z),'top10_net_ev':p.execution_net_ev_12h.mean(),'execution_rank_ic':z.mapped_score.corr(z.execution_net_ev_12h,method='spearman'),'top10_rows':len(p),'availability_selected':p.trajectory_available.mean(),'selected_asset_hhi':p.__symbol__.value_counts(normalize=True).pow(2).sum()});selected.append(p[['candidate_id','__ts__','__symbol__','side_name','trajectory_available','execution_net_ev_12h','mapped_score']].assign(arm=a))
  for kind,key in [('week',z.__ts__.dt.strftime('%G-W%V')),('month',z.__ts__.dt.strftime('%Y-%m'))]:
   for k,g in z.groupby(key):
    pp=g[g.selected_global_top10];per.append({'arm':a,'period_type':kind,'period':k,'mean_net_ev':pp.execution_net_ev_12h.mean(),'global_selected_rows':len(pp)})
  for s,g in z.groupby('side_name'):
   pp=g[g.selected_global_top10];side.append({'arm':a,'side_name':s,'top10_net_ev':pp.execution_net_ev_12h.mean(),'execution_rank_ic':g.mapped_score.corr(g.execution_net_ev_12h,method='spearman')})
  avail.append(z.groupby('trajectory_available').apply(lambda g: pd.Series({'arm':a,'rows':len(g),'rank_ic':g.mapped_score.corr(g.execution_net_ev_12h,method='spearman'),'selected_net_ev':g.loc[g.selected_global_top10,'execution_net_ev_12h'].mean()})).reset_index())
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  r=pd.DataFrame(rows);p=pd.DataFrame(per);q=[]
  for a,g in p.groupby('arm'):
   d={'arm':a}
   for k in ['week','month']:
    x=g[g.period_type.eq(k)].mean_net_ev;d[k+'_net_ev_q10']=x.quantile(.1);d[k+'_net_ev_q50']=x.quantile(.5);d['latest_'+k+'_net_ev']=g[g.period_type.eq(k)].sort_values('period').iloc[-1].mean_net_ev;d['worst_'+k+'_net_ev']=x.min()
   q.append(d)
  r=r.merge(pd.DataFrame(q),on='arm');sel=pd.concat(selected,ignore_index=True);base=set(sel[sel.arm.eq('baseline_existing_transition_control')].candidate_id);turn=[]
  for a,g in sel.groupby('arm'):
   ids=set(g.candidate_id);turn.append({'arm':a,'selected_rows':len(ids),'overlap_with_baseline_rows':len(ids&base),'replacement_rate_vs_baseline':1-len(ids&base)/len(base)})
  r=r.merge(pd.DataFrame(turn),on='arm');r.to_csv(stage/'metrics_summary.csv',index=False);p.to_parquet(stage/'period_metrics.parquet',index=False);pd.DataFrame(side).to_parquet(stage/'side_metrics.parquet',index=False);pd.concat(avail).to_csv(stage/'availability_metrics.csv',index=False);sel.to_parquet(stage/'selected_candidates.parquet',index=False);contract=json.loads((PRE/'contract.json').read_text());contract['execution']='completed exactly as preregistered';dump(stage/'contract.json',contract);files=[x for x in stage.iterdir() if x.is_file()];m={'schema':'trajectory_missingness_identical_row_ablation_v1','status':'SEALED_PREREGISTERED_TRAJECTORY_MISSINGNESS_ABLATION_NON_PROMOTION','promotion_eligible':False,'contract':contract,'input_rows':{'historical':len(h),'frozen_2026':len(f),'frozen_2026_trajectory_available':int(f.trajectory_available.sum())},'outputs_sha256':{x.name:sha(x) for x in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
