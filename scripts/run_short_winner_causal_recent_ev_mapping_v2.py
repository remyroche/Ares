#!/usr/bin/env python3
"""Immutable short-winner score ledger plus fixed causal recent-EV maps."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_bounded_robust_auxiliary_contribution_ablation as base
from scripts import run_bounded_short_conditional_payoff_ablation as short
from scripts.run_bounded_side_local_support_composition import strict_mae
from scripts.correct_bounded_side_local_support_composition_ties import bound
WIN=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2';MAE=short.MAE;POOL=2000;SIDE=1000;LAMBDA=500.
def inputs(a):
 x=base.load(a);m,s=strict_mae(a.mae)
 if m is None:raise RuntimeError(s['status'])
 m['__ts__']=pd.to_datetime(m['__ts__'],utc=True);return x.merge(m,on=list(base.ID),validate='one_to_one')
def march_ledger(x):
 d,_,_=base.reconstruct(x);d=d[np.isfinite(d.robust_decomposed)].copy().reset_index(drop=True);d['peak']=d.pred_peak_mfe_12h_atr__p_hit*d.pred_peak_mfe_12h_atr__conditional_mean;feat=short.F+['peak','pred_future_slope_atr_per_hour__diagnostic'];out=np.full(len(d),np.nan);fold=np.full(len(d),'',dtype=object);days=np.array(sorted(d[base.TIME].dt.floor('D').unique()))
 for cut in [days[int(len(days)*q)] for q in (.4,.6,.8)]:
  vi=(d[base.TIME]>=cut)&(d[base.TIME]<cut+pd.Timedelta(days=6));ti=(d[base.TIME]<cut)&(d[base.END]<cut);tr=d[ti&d.side_name.eq('short')];va=d[vi&d.side_name.eq('short')]
  if len(va):*_,v=short.fit_decomp(tr,va,feat,2.);out[va.index]=v;fold[va.index]=str(cut)
 d['raw_score']=np.where(d.side_name.eq('short'),out,d.robust_decomposed);d['fold_train_cutoff_utc']=np.where(d.side_name.eq('short'),fold,'preexisting_strict_residual_oof');d['validation_interval_utc']=np.where(d.side_name.eq('short'),fold+'..+6d','preexisting_march_outer_oof');d['score_available_utc']=d[base.TIME];d['is_candidate_head_oof']=d.side_name.eq('short');d['upstream_outer_oof']=True;d['is_forward_oos']=False;d['ledger_stage']='march_inner_chronological_oof';return d[np.isfinite(d.raw_score)].copy()
def daily(history,evals,side_residual):
 n=len(evals);score=np.full(n,np.nan);eligible=np.zeros(n,bool);status=np.full(n,'unmapped_weak_pooled',object);pn=np.zeros(n,int);sn=np.zeros(n,int)
 for day in sorted(evals[base.TIME].dt.floor('D').unique()):
  ix=evals[base.TIME].dt.floor('D').eq(day).to_numpy();h=history[(history[base.TIME]<day)&(history[base.END]<day)&(history[base.END]>=day-pd.Timedelta(days=21))]
  pn[ix]=len(h)
  if len(h)<POOL or h.raw_score.nunique()<2:continue
  iso=IsotonicRegression(out_of_bounds='clip').fit(h.raw_score,h[base.Y]);p=iso.predict(evals.loc[ix,'raw_score']);eligible[ix]=True;status[ix]='pooled_anchor'
  if side_residual:
   for side in ('long','short'):
    local=h.side_name.eq(side).to_numpy();rows=int(local.sum());loc=ix & evals.side_name.eq(side).to_numpy();sn[loc]=rows
    if rows>=SIDE:
     sideiso=IsotonicRegression(out_of_bounds='clip').fit(h.loc[local,'raw_score'],h.loc[local,base.Y]);target=evals.loc[loc,'raw_score'];p[evals.loc[ix,'side_name'].to_numpy()==side]+=rows/(rows+LAMBDA)*(sideiso.predict(target)-iso.predict(target));status[loc]='pooled_plus_side_residual'
    else:status[loc]='pooled_zero_side_residual'
  score[ix]=p
 return score,eligible,status,pn,sn
def report(frame,col,name):
 rows=[]
 for f in (.01,.05,.1,.2):
  q=base.order(frame,col,f);b=bound(frame,col,f);b.update({'map':name,'net_bps':float(q[base.Y].mean()*1e4),'gross_bps':float(q.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(q.execution_cost_return.mean()*1e4),'bias_bps':float((q[col]-q[base.Y]).mean()*1e4),'mae_bps':float(np.abs(q[col]-q[base.Y]).mean()*1e4)});rows.append(b)
 return rows
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 winner_manifest=json.loads((a.winner/'manifest.json').read_text());assert winner_manifest['frozen_winner']['arm']=='B_peak_slope' and float(winner_manifest['frozen_winner']['short_tail_weight'])==2.0
 x=inputs(a);march=march_ledger(x);apr=pd.read_parquet(a.winner/'april_confirmation_predictions.parquet');apr['__ts__']=pd.to_datetime(apr['__ts__'],utc=True);apr[base.TIME]=pd.to_datetime(apr[base.TIME],utc=True);apr[base.END]=pd.to_datetime(apr[base.END],utc=True);apr['raw_score']=apr['raw_score'].astype(float);apr['score_available_utc']=apr[base.TIME];apr['ledger_stage']='april_frozen_forward';apr['fold_train_cutoff_utc']='2025-04-01T01:00:00+00:00';apr['validation_interval_utc']='2025-04-01..2025-05-01';apr['is_candidate_head_oof']=False;apr['upstream_outer_oof']=True;apr['is_forward_oos']=True
 # join exact realised accounting only for evaluation, never map inputs.
 econ=x[x.candidate_month.eq('2025-04')][list(base.ID)+['execution_gross_ev_12h','execution_cost_return','score_base_alpha','score_residual_expected_ev','direct_q25_return']];econ['__ts__']=pd.to_datetime(econ['__ts__'],utc=True);apr=apr.merge(econ,on=list(base.ID),validate='one_to_one')
 for field in ('execution_gross_ev_12h','execution_cost_return'):
  if field+'_x' in apr:
   apr[field]=apr[field+'_x'];apr=apr.drop(columns=[field+'_x',field+'_y'])
 for field in ('score_base_alpha','score_residual_expected_ev','direct_q25_return'):
  if field+'_y' in apr:
   apr[field]=apr[field+'_y'];apr=apr.drop(columns=[field+'_x',field+'_y'])
 iso=IsotonicRegression(out_of_bounds='clip').fit(march.raw_score,march[base.Y]);apr['frozen_march_isotonic']=iso.predict(apr.raw_score)
 full_history=pd.concat([march,apr],ignore_index=True)
 anchor=daily(full_history,apr,False);shrink=daily(full_history,apr,True)
 for name,val in [('anchor_21d',anchor),('anchor_side_shrink',shrink)]:apr[name],apr[name+'_eligible'],apr[name+'_status'],apr[name+'_pooled_rows'],apr[name+'_side_rows']=val
 rows=[];latest=[];side=[];asset=[]
 for name,col,eligible in [('raw','raw_score',np.ones(len(apr),bool)),('frozen_march_isotonic','frozen_march_isotonic',np.ones(len(apr),bool)),('anchor_21d','anchor_21d',apr.anchor_21d_eligible.to_numpy()),('anchor_side_shrink','anchor_side_shrink',apr.anchor_side_shrink_eligible.to_numpy())]:
  z=apr.loc[eligible].copy();rows+=report(z,col,name)
  for f in (.01,.05,.1,.2):
   q=base.order(z,col,f);week=z.__ts__.max().floor('D')-pd.Timedelta(days=6);w=base.order(z[z.__ts__>=week],col,f);latest.append({'map':name,'top_fraction':f,'coverage_rows':len(z),'coverage_fraction':len(z)/len(apr),'latest_week_net_bps':float(w.execution_net_ev_12h.mean()*1e4)})
   if f==.1:
    for k,v in q.groupby('side_name'):side.append({'map':name,'side_name':k,'rows':len(v),'share':len(v)/len(q),'net_bps':float(v.execution_net_ev_12h.mean()*1e4)})
    for k,v in q.groupby('__symbol__'):asset.append({'map':name,'__symbol__':k,'rows':len(v),'share':len(v)/len(q),'net_bps':float(v.execution_net_ev_12h.mean()*1e4)})
 controls=[]
 for c in ['score_base_alpha','score_residual_expected_ev','direct_q25_return']:
  for f in (.01,.05,.1,.2):controls.append({'control':c,'top_fraction':f,'net_bps':float(base.order(apr,c,f).execution_net_ev_12h.mean()*1e4)})
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));march.to_parquet(st/'march_inner_chronological_oof_score_ledger.parquet',index=False);apr.to_parquet(st/'april_frozen_forward_score_ledger_and_maps.parquet',index=False);pd.DataFrame(rows).to_csv(st/'metrics.csv',index=False);pd.DataFrame(latest).to_csv(st/'latest_week.csv',index=False);pd.DataFrame(side).to_csv(st/'side_top10.csv',index=False);pd.DataFrame(asset).to_csv(st/'asset_top10.csv',index=False);pd.DataFrame(controls).to_csv(st/'identical_id_controls.csv',index=False);outs={p.name:base.hs(p) for p in st.iterdir() if p.is_file()};man={'schema':'short_winner_causal_recent_ev_mapping_v2','status':'RESEARCH_ONLY_NO_REPLAY','contract':{'ranker':'frozen April winner; no ranker refit','ledger':'candidate keys, fold cutoff, score availability, label_end, strict March OOF and untouched April forward','pooled_support':POOL,'side_support':SIDE,'side_residual_weight':'n_side/(n_side+500)','weak_pooled':'mapped score NaN/status unmapped_weak_pooled/excluded mapped topK','weak_side':'pooled anchor plus exactly zero residual','selection':'pooled global topK no quota','actions':'excluded'},'march_rows':len(march),'april_rows':len(apr),'inputs':{'winner_manifest':base.hs(a.winner/'manifest.json'),'source':base.hs(a.source),'runner':base.hs(Path(__file__))},'outputs_sha256':outs};base.wj(st/'manifest.json',man);(st/'manifest.sha256').write_text(base.hs(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=base.SRC);p.add_argument('--peak',type=Path,default=base.PEAK);p.add_argument('--slope',type=Path,default=base.SLOPE);p.add_argument('--mae',type=Path,default=MAE);p.add_argument('--winner',type=Path,default=WIN);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,default=str))
