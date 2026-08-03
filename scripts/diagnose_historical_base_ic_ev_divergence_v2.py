#!/usr/bin/env python3
"""Exact native-base Feb--Apr score versus 12h execution divergence audit."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd
ID=["candidate_id","side_name","__symbol__","__ts__"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 t=p.with_name(f'.{p.name}.{os.getpid()}.partial');t.write_text(json.dumps(x,indent=2,default=str)+'\n');os.replace(t,p)
def ic(q,col):
 v=q.old_score.corr(q[col],method='spearman');return float(v) if np.isfinite(v) else None
def top(q, frac=.1):
 n=int(np.ceil(len(q)*frac));s=q.nlargest(n,'old_score');go=set(q.nlargest(n,'execution_gross_ev_12h').candidate_id);no=set(q.nlargest(n,'execution_net_ev_12h').candidate_id);p=set(s.candidate_id)
 asset=s.__symbol__.value_counts(normalize=True);exit_detail=[]
 for name,g in s.groupby('execution_exit_reason'):
  exit_detail.append({'exit_reason':str(name),'share':float(len(g)/len(s)),'net_bps':float(g.execution_net_ev_12h.mean()*1e4),'mfe_bps':float(g.execution_mfe_return_12h.mean()*1e4),'mae_bps':float(g.execution_mae_return_12h.mean()*1e4),'exit_minute_mean':float(g.execution_exit_minute.mean())})
 return {'rows':len(s),'gross_bps':float(s.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(s.execution_cost_return.mean()*1e4),'net_bps':float(s.execution_net_ev_12h.mean()*1e4),'median_net_bps':float(s.execution_net_ev_12h.median()*1e4),'positive_net_precision':float(s.execution_net_ev_12h.gt(0).mean()),'gross_oracle_recall':float(len(p&go)/len(go)),'net_oracle_recall':float(len(p&no)/len(no)),'side_capacity':s.side_name.value_counts().to_dict(),'asset_count':int(s.__symbol__.nunique()),'asset_top_share':float(asset.iloc[0]),'asset_hhi':float((asset**2).sum()),'exit_composition':s.execution_exit_reason.value_counts(normalize=True).to_dict(),'exit_conditional_path':exit_detail,'mfe_mean':float(s.execution_mfe_return_12h.mean()),'mae_mean':float(s.execution_mae_return_12h.mean()),'exit_minute_mean':float(s.execution_exit_minute.mean()),'exit_horizon_composition':s.exit_time_bucket.value_counts(normalize=True).to_dict()}
def deciles(q):
 d=pd.qcut(q.old_score.rank(method='first'),10,labels=False)+1
 return q.assign(decile=d).groupby('decile').agg(rows=('old_score','size'),native_target=('target24','mean'),gross=('execution_gross_ev_12h','mean'),cost=('execution_cost_return','mean'),net=('execution_net_ev_12h','mean'),positive_net=('execution_net_ev_12h',lambda x:float((x>0).mean()))).reset_index().to_dict('records')
def strata(q,cols):
 out=[]
 for c in cols:
  if c not in q: continue
  z=pd.qcut(q[c].rank(method='first'),4,labels=False)+1
  for k,g in q.assign(_s=z).groupby('_s'):
   out.append({'feature':c,'quartile':int(k),'rows':len(g),'ic_native_target':ic(g,'target24'),'ic_gross':ic(g,'execution_gross_ev_12h'),'ic_net':ic(g,'execution_net_ev_12h'),'global_within_stratum_top10_net_bps':top(g)['net_bps']})
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--attribution-root',type=Path,required=True);a.add_argument('--context-root',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);z=a.parse_args();part=z.output_root.with_name(z.output_root.name+'.partial')
 if z.output_root.exists() or part.exists():raise FileExistsError(z.output_root)
 m=json.loads((z.attribution_root/'manifest.json').read_text());cm=json.loads((z.context_root/'manifest.json').read_text())
 if m.get('schema')!='native12_execution_ev_failure_attribution_v4' or cm.get('status')!='IMMUTABLE_PREENTRY_ONLY_INPUT_PANEL':raise ValueError('requires v4 attribution and repaired v3 context')
 for n,h in m['outputs_sha256'].items():
  if sha(z.attribution_root/n)!=h:raise ValueError('attribution hash mismatch '+n)
 for n,h in cm['outputs_sha256'].items():
  if sha(z.context_root/n)!=h:raise ValueError('context hash mismatch '+n)
 x=pd.read_parquet(z.attribution_root/'joined_frozen_attribution_rows.parquet');ctx=pd.read_parquet(z.context_root/'panel.parquet',columns=[*ID,*cm['feature_columns']]);x=x.merge(ctx,on=ID,how='left',validate='one_to_one',indicator='_ctx')
 if len(x)!=509868:raise ValueError('identity mismatch')
 identity_sha=hashlib.sha256(x[ID].sort_values(ID).to_csv(index=False,lineterminator='\n').encode()).hexdigest()
 features=['range_24h_pct','__meta_raw__volatility_zscore','trend_r2_24','jump_intensity','preentry_transition__range_24h_pct__delta_3h']
 report={'schema':'historical_base_ic_ev_divergence_v4','status':'diagnostic_non_promotion','source_audit':{'attribution_manifest_sha256':sha(z.attribution_root/'manifest.json'),'context_manifest_sha256':sha(z.context_root/'manifest.json'),'rows':len(x),'identity_sha256':identity_sha,'native_score':'old_score = canonical base_oof_score','native_target':'target24 = __first_touch_target_soft__','execution':'exact 12h gross - realized cost = net','mapping_status':'raw frozen base-score diagnosis only; no mapped-vs-raw claim'},'cost_ic_interpretation':'Spearman is rank-only. When cost is nearly constant, affine, or monotonically co-moves with gross/net, its ranks are tied/aligned; similar gross/cost/net Spearman does not mean the score predicts variable trading cost. Mean economics and cost dispersion are reported separately.','months':{}}
 for month,q in x.groupby('month',sort=True):
  coverage=float(q._ctx.eq('both').mean());by_side={side:{'rows':len(g),'rank_ic':{'native_target24':ic(g,'target24'),'native_target12':ic(g,'target12'),'gross':ic(g,'execution_gross_ev_12h'),'cost':ic(g,'execution_cost_return'),'net':ic(g,'execution_net_ev_12h')},'top10_local_diagnostic':top(g)} for side,g in q.groupby('side_name')};report['months'][month]={'rows':len(q),'context_coverage':coverage,'rank_ic':{'native_target24':ic(q,'target24'),'native_target12':ic(q,'target12'),'gross':ic(q,'execution_gross_ev_12h'),'cost':ic(q,'execution_cost_return'),'net':ic(q,'execution_net_ev_12h')},'by_side':by_side,'cost_distribution':{'mean_bps':float(q.execution_cost_return.mean()*1e4),'std_bps':float(q.execution_cost_return.std()*1e4),'unique_rounded_1bp':int(q.execution_cost_return.round(4).nunique())},'pooled_global_top10':top(q),'pooled_global_quantiles':[{**{'quantile':f'top_{p:.0%}'},**top(q,p)} for p in (.01,.05,.1,.2)],'native_decile_monotonicity':deciles(q),'composition':{'side':q.side_name.value_counts(normalize=True).to_dict(),'assets':int(q.__symbol__.nunique()),'asset_count':int(q.__symbol__.nunique())},'failure_attribution':{'exit_all':q.execution_exit_reason.value_counts(normalize=True).to_dict(),'mfe_all_mean':float(q.execution_mfe_return_12h.mean()),'mae_all_mean':float(q.execution_mae_return_12h.mean()),'exit_horizon_all':q.exit_time_bucket.value_counts(normalize=True).to_dict()},'regime_strata':strata(q.loc[q._ctx.eq('both')],features) if coverage else []}
 expected={'2025-02':.15498445540493136,'2025-03':.161872,'2025-04':.225896};observed={m:report['months'][m]['by_side']['long']['rank_ic']['native_target24'] for m in expected};report['quoted_long_native_target_ic_reproduction']={'expected':expected,'observed':observed,'status':'reproduced' if all(abs(observed[m]-expected[m])<.001 for m in expected) else 'not_reproduced'}
 part.mkdir(parents=True);rp=part/'report.json';write(rp,report);manifest={'schema':'historical_base_ic_ev_divergence_v4_manifest','status':'diagnostic_non_promotion','runner':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve())},'sources':{'attribution_manifest':{'path':str(z.attribution_root/'manifest.json'),'sha256':sha(z.attribution_root/'manifest.json')},'context_manifest':{'path':str(z.context_root/'manifest.json'),'sha256':sha(z.context_root/'manifest.json')}},'output_sha256':{'report.json':sha(rp)}};write(part/'manifest.json',manifest);part.replace(z.output_root)
if __name__=='__main__':main()
