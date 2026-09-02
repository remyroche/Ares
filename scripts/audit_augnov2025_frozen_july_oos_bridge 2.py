#!/usr/bin/env python3
"""Seal integrity and base-versus-residual economics for the Aug--Nov bridge."""
from __future__ import annotations
import hashlib,json,math,os,shutil,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; BRIDGE=ROOT/'data_perp/artifacts/augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1';OUT=ROOT/'data_perp/artifacts/augnov2025_frozen_july_oos_bridge_validation_economics_20260730_v1';TOP=.10
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def rank(a,b):return float(a.corr(b,method='spearman'))
def evaluate(x,name):
 w=x.sort_values([name,'candidate_id'],ascending=[False,True],kind='stable').copy();w['selected_global_top10']=False;w.loc[w.index[:math.ceil(len(w)*TOP)],'selected_global_top10']=True;s=w[w.selected_global_top10];rows=[]
 for kind,key in [('week',w.__ts__.dt.strftime('%G-W%V')),('month',w.__ts__.dt.strftime('%Y-%m'))]:
  for period,z in w.groupby(key,observed=True,sort=True):
   p=z[z.selected_global_top10];rows.append({'score':name,'period_type':kind,'period':period,'candidate_rows':len(z),'global_selected_rows':len(p),'rank_ic':rank(z[name],z.execution_net_ev_12h),'mean_net_ev':p.execution_net_ev_12h.mean(),'mean_gross_ev':p.execution_gross_ev_12h.mean(),'mean_cost':p.execution_cost_return.mean(),'hit_rate':p.execution_net_ev_12h.gt(0).mean()})
 q=pd.DataFrame(rows);summary={'score':name,'candidate_rows':len(w),'top10_rows':len(s),'rank_ic':rank(w[name],w.execution_net_ev_12h),'top10_net_ev':s.execution_net_ev_12h.mean(),'top10_gross_ev':s.execution_gross_ev_12h.mean(),'top10_cost':s.execution_cost_return.mean(),'top10_hit_rate':s.execution_net_ev_12h.gt(0).mean()}
 for k in ['week','month']:
  z=q[q.period_type.eq(k)];summary[f'{k}_net_ev_q10']=z.mean_net_ev.quantile(.1);summary[f'{k}_net_ev_q50']=z.mean_net_ev.quantile(.5)
 sides=[]
 for side,z in w.groupby('side_name',observed=True):
  p=z[z.selected_global_top10];sides.append({'score':name,'side_name':side,'candidate_rows':len(z),'global_selected_rows':len(p),'rank_ic':rank(z[name],z.execution_net_ev_12h),'top10_net_ev':p.execution_net_ev_12h.mean(),'top10_gross_ev':p.execution_gross_ev_12h.mean(),'top10_cost':p.execution_cost_return.mean(),'top10_hit_rate':p.execution_net_ev_12h.gt(0).mean()})
 return summary,q,pd.DataFrame(sides)
def run(bridge=BRIDGE,output=OUT):
 output=Path(output);bridge=Path(bridge)
 if output.exists():raise RuntimeError(output)
 m=json.loads((bridge/'manifest.json').read_text());marker=(bridge/'manifest.sha256').read_text().split()[0]
 if marker!=sha(bridge/'manifest.json') or m.get('status')!='SEALED_COMMON30_FROZEN_JULY_OOS_SCORE_BRIDGE_NON_PROMOTION':raise RuntimeError('unsealed bridge')
 p=bridge/'oos_predictions.parquet'
 if m['outputs_sha256'][p.name]!=sha(p):raise RuntimeError('output hash mismatch')
 x=pd.read_parquet(p);x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['execution_label_end_utc']=pd.to_datetime(x.execution_label_end_utc,utc=True)
 checks={'rows':len(x),'unique_candidate_ids':x.candidate_id.nunique(),'duplicates':int(x.candidate_id.duplicated().sum()),'hourly_timestamps':bool((x.__ts__.astype('int64')%pd.Timedelta(hours=1).value==0).all()),'strict_oos_cutoff_base':bool(pd.to_datetime(x.base_score_fit_cutoff_utc,utc=True).eq(pd.Timestamp('2025-08-01',tz='UTC')).all()),'strict_oos_cutoff_residual':bool(pd.to_datetime(x.residual_score_fit_cutoff_utc,utc=True).eq(pd.Timestamp('2025-08-01',tz='UTC')).all()),'all_base_oos':bool(x.base_is_oos.all()),'all_residual_oos':bool(x.residual_is_oos.all()),'label_end_after_decision':bool(x.execution_label_end_utc.gt(x.__ts__).all()),'by_month':x.__ts__.dt.strftime('%Y-%m').value_counts().sort_index().to_dict(),'by_side':x.side_name.value_counts().to_dict()}
 if checks['rows']!=175680 or checks['duplicates'] or not all(v for k,v in checks.items() if isinstance(v,bool)):raise RuntimeError(f'bridge validation failed: {checks}')
 rows=[];periods=[];sides=[]
 for score in ['score_base_alpha','score_residual_expected_ev']:
  a,b,c=evaluate(x,score);rows.append(a);periods.append(b);sides.append(c)
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  pd.DataFrame(rows).to_csv(stage/'metrics_summary.csv',index=False);pd.concat(periods).to_parquet(stage/'period_metrics.parquet',index=False);pd.concat(sides).to_parquet(stage/'side_metrics.parquet',index=False);dump(stage/'validation.json',checks);files=[z for z in stage.iterdir() if z.is_file()];manifest={'schema':'augnov2025_frozen_july_oos_bridge_validation_economics_v1','status':'SEALED_VALIDATED_COMMON30_OOS_BRIDGE_ECONOMICS_NON_PROMOTION','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','inputs':{str((bridge/'manifest.json').resolve()):sha(bridge/'manifest.json'),str(p.resolve()):sha(p)},'outputs_sha256':{z.name:sha(z) for z in files}};dump(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
