#!/usr/bin/env python3
"""Fixed pre-August context-arm OOS scoring on sealed Aug--Nov common-30 rows."""
from __future__ import annotations
import hashlib,json,math,os,shutil,tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_final_identical_row_regime_stack_gam_ablation as final
IDENTITY=('candidate_id','__ts__','__symbol__','side_name');TARGET=final.TARGET;RES=final.RESIDUAL;CUT=pd.Timestamp('2025-08-01',tz='UTC');END=pd.Timestamp('2025-12-01',tz='UTC');TOP=.10
SIDECARES=ROOT/'data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1/manifest.json';HISTORY=ROOT/'data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet';JULY=ROOT/'data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1';AUGNOV=ROOT/'data_perp/artifacts/augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1';OUT=ROOT/'data_perp/artifacts/augnov2025_common30_fixed_preaug_context_oos_extension_20260730_v2'
class Err(RuntimeError):pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def sealed(root,schema,status):
 root=Path(root);m=root/'manifest.json';mark=root/'manifest.sha256'
 if not m.is_file() or not mark.is_file() or mark.read_text().split()[0]!=sha(m):raise Err(f'unsealed {root}')
 d=json.loads(m.read_text())
 if d.get('schema')!=schema or d.get('status')!=status:raise Err(f'wrong sealed contract {root}')
 return d
def join_context(scores,context,phase):
 before=scores[list(IDENTITY)].sort_values(list(IDENTITY),kind='stable').reset_index(drop=True);o=scores.merge(context,left_on='__ts__',right_on='source_utc',how='left',validate='many_to_one',sort=False);after=o[list(IDENTITY)].sort_values(list(IDENTITY),kind='stable').reset_index(drop=True)
 if len(o)!=len(scores) or not before.equals(after) or o.source_utc.isna().any():raise Err(f'{phase}: context join changes identity')
 required=[*final.REGIME,*final.TRANSITION];avail=o[required].notna().all(axis=1)&o.bocpd_regime_available.astype(bool)&o.lgbm_transition_available.astype(bool)
 if not avail.all():raise Err(f'{phase}: unavailable context')
 for suffix in ('bocpd','lgbm'):
  if not o[f'provenance_partition_{suffix}'].eq('blocked_oof_2022_2025').all():raise Err(f'{phase}: non-blocked context')
  end=pd.to_datetime(o[f'train_end_exclusive_utc_{suffix}'],utc=True);resolved=pd.to_datetime(o[f'fit_label_resolution_max_utc_{suffix}'],utc=True)
  if end.isna().any() or resolved.isna().any() or not resolved.lt(end).all() or not end.le(CUT).all():raise Err(f'{phase}: context fit not pre-Aug')
 return o.drop(columns='source_utc')
def rank(a,b):return float(a.corr(b,method='spearman'))
def evaluate(x,arm):
 w=x.sort_values(['raw_score','candidate_id'],ascending=[False,True],kind='stable').copy();w['selected_global_top10']=False;w.loc[w.index[:math.ceil(len(w)*TOP)],'selected_global_top10']=True;s=w[w.selected_global_top10]
 summary={'arm':arm,'candidate_rows':len(w),'top10_rows':len(s),'execution_rank_ic':rank(w.raw_score,w[TARGET]),'top10_net_ev':s[TARGET].mean(),'top10_gross_ev':s.execution_gross_ev_12h.mean(),'top10_cost':s.execution_cost_return.mean(),'top10_hit_rate':s[TARGET].gt(0).mean()};rows=[]
 for kind,key in [('week',w.__ts__.dt.strftime('%G-W%V')),('month',w.__ts__.dt.strftime('%Y-%m'))]:
  for period,z in w.groupby(key,observed=True,sort=True):
   p=z[z.selected_global_top10];rows.append({'arm':arm,'period_type':kind,'period':period,'candidate_rows':len(z),'global_selected_rows':len(p),'execution_rank_ic':rank(z.raw_score,z[TARGET]),'mean_net_ev':p[TARGET].mean(),'mean_gross_ev':p.execution_gross_ev_12h.mean(),'mean_cost':p.execution_cost_return.mean(),'hit_rate':p[TARGET].gt(0).mean()})
 period_frame=pd.DataFrame(rows)
 for kind in ['week','month']:
  z=period_frame[period_frame.period_type.eq(kind)];summary[f'{kind}_net_ev_q10']=z.mean_net_ev.quantile(.1);summary[f'{kind}_net_ev_q50']=z.mean_net_ev.quantile(.5);summary[f'latest_{kind}']=z.period.max();summary[f'latest_{kind}_net_ev']=z.sort_values('period').iloc[-1].mean_net_ev;summary[f'worst_{kind}']=z.loc[z.mean_net_ev.idxmin()].period;summary[f'worst_{kind}_net_ev']=z.mean_net_ev.min()
 sides=[]
 for side,z in w.groupby('side_name',observed=True):
  selected_side=z[z.selected_global_top10];sides.append({'arm':arm,'side_name':side,'candidate_rows':len(z),'global_selected_rows':len(selected_side),'execution_rank_ic':rank(z.raw_score,z[TARGET]),'top10_net_ev':selected_side[TARGET].mean(),'top10_gross_ev':selected_side.execution_gross_ev_12h.mean(),'top10_cost':selected_side.execution_cost_return.mean(),'top10_hit_rate':selected_side[TARGET].gt(0).mean()})
 return summary,period_frame,pd.DataFrame(sides),w
def run(sidecars=SIDECARES,historical=HISTORY,july_root=JULY,augnov_root=AUGNOV,output=OUT):
 output=Path(output)
 if output.exists():raise Err(output)
 _,rp,tp=final._load_manifest(Path(sidecars));context=final._hourly_context(rp,tp)
 h=final._join(final._verified_scores(Path(historical),role='historical'),context,role='historical')
 jm=sealed(july_root,'july2025_common30_final_base_residual_oof_bridge_v1','SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION');jpath=Path(july_root)/'oof_predictions.parquet'
 jc=json.loads((Path(july_root)/'bridge_contract.json').read_text())
 if jc.get('status')!='SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION' or jc.get('outputs',{}).get(jpath.name)!=sha(jpath):raise Err('July scores checksum')
 j=pd.read_parquet(jpath,columns=[*IDENTITY,TARGET,RES,'execution_label_end_utc','residual_is_oof']);j['__ts__']=pd.to_datetime(j.__ts__,utc=True);j['execution_label_end_utc']=pd.to_datetime(j.execution_label_end_utc,utc=True)
 if len(j)!=44640 or not j.residual_is_oof.all():raise Err('July OOF')
 j=join_context(j,context,'July train')
 am=sealed(augnov_root,'augnov2025_common30_frozen_july_base_residual_oos_bridge_v1','SEALED_COMMON30_FROZEN_JULY_OOS_SCORE_BRIDGE_NON_PROMOTION');apath=Path(augnov_root)/'oos_predictions.parquet'
 if am['outputs_sha256'][apath.name]!=sha(apath):raise Err('AugNov scores checksum')
 t=pd.read_parquet(apath);t['__ts__']=pd.to_datetime(t.__ts__,utc=True);t['execution_label_end_utc']=pd.to_datetime(t.execution_label_end_utc,utc=True)
 if len(t)!=175680 or t.candidate_id.duplicated().any() or not t.residual_is_oos.all() or not t.__ts__.between(CUT,END-pd.Timedelta(hours=1)).all():raise Err('AugNov OOS identity')
 t=join_context(t,context,'AugNov test')
 train=pd.concat([h,j],ignore_index=True,sort=False);train=train[train.execution_label_end_utc.lt(CUT)].copy()
 if train.empty or not train.execution_label_end_utc.lt(CUT).all() or train.__ts__.ge(CUT).any():raise Err('future label in context fit')
 arms=[]
 for family,place in [('lgbm','residual_trust'),('gam','additive_bounded_gam')]:
  for ctx in ['regime','transition','combined']:arms.append(final.Arm(f'{place}_{ctx}',place,ctx,TARGET,family))
 summaries=[];periods=[];sides=[];scores=[];audit=[]
 for n,arm in enumerate(arms):
  pieces=[]
  for side,local in t.groupby('side_name',observed=True,sort=True):
   fit=train[train.side_name.eq(side)].copy();raw,meta=final._predict(fit,local,arm,20250800+n*31+(side=='short'));pieces.append(local.assign(arm=arm.name,raw_score=raw));audit.append({'arm':arm.name,'side_name':side,'fit_rows':len(fit),'fit_label_end_max':fit.execution_label_end_utc.max(),'fit_labels_before_aug':bool(fit.execution_label_end_utc.lt(CUT).all()),'evaluation_rows':len(local),'evaluation_start':local.__ts__.min(),'evaluation_end':local.__ts__.max(),**meta})
  scored=pd.concat(pieces,ignore_index=True);a,b,c,d=evaluate(scored,arm.name);summaries.append(a);periods.append(b);sides.append(c);scores.append(d[[*IDENTITY,'execution_label_end_utc',TARGET,'execution_gross_ev_12h','execution_cost_return','arm','raw_score','selected_global_top10']])
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  pd.DataFrame(summaries).to_csv(stage/'metrics_summary.csv',index=False);pd.concat(periods).to_parquet(stage/'period_metrics.parquet',index=False);pd.concat(sides).to_parquet(stage/'side_metrics.parquet',index=False);pd.concat(scores).to_parquet(stage/'augnov_raw_context_scores.parquet',index=False);dump(stage/'fit_audit.json',audit)
  contract={'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','arms':[a.__dict__ for a in arms],'training':'side-local frozen final context architectures; compatible historical OOF plus July common30 OOF rows with execution labels resolved strictly before 2025-08-01','assessment':'sealed Aug-Nov common30 OOS scores; one pooled global raw-score top10 per arm, period tables only decompose fixed membership','no_hpo_or_feature_selection':True,'no_2026_outcomes':True,'scope_limitation':'common30 only; no causal EV map or promotion is produced','supersedes_invalid':'augnov2025_common30_fixed_preaug_context_oos_extension_20260730_v1 (period_metrics contained selected rows, not period aggregates)'};dump(stage/'contract.json',contract);files=[z for z in stage.iterdir() if z.is_file()];manifest={'schema':'augnov2025_common30_fixed_preaug_context_oos_extension_v2','status':'SEALED_FIXED_PREAUG_CONTEXT_OOS_EXTENSION_NON_PROMOTION','promotion_eligible':False,'inputs':{str(Path(sidecars).resolve()):sha(sidecars),str(Path(historical).resolve()):sha(historical),str((Path(july_root)/'manifest.json').resolve()):sha(Path(july_root)/'manifest.json'),str((Path(augnov_root)/'manifest.json').resolve()):sha(Path(augnov_root)/'manifest.json')},'contract':contract,'outputs_sha256':{z.name:sha(z) for z in files}};dump(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
