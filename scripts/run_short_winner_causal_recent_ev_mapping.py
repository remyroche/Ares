#!/usr/bin/env python3
"""Causal daily recent-EV maps over the frozen short-conversion winner."""
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
WIN=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2';MAE=short.MAE
GRID=((0.,300),(0.5,300),(1.,300),(0.5,600),(1.,600))
def load(a):
 x=base.load(a);m,status=strict_mae(a.mae)
 if m is None:raise RuntimeError(status['status'])
 m['__ts__']=pd.to_datetime(m['__ts__'],utc=True);x=x.merge(m,on=list(base.ID),validate='one_to_one');return x
def march_scores(x):
 d,c,_=base.reconstruct(x);d=d[np.isfinite(d.robust_decomposed)].copy().reset_index(drop=True);d['peak_contribution']=d.pred_peak_mfe_12h_atr__p_hit*d.pred_peak_mfe_12h_atr__conditional_mean
 feat=short.F+['peak_contribution','pred_future_slope_atr_per_hour__diagnostic'];out=np.full(len(d),np.nan);days=np.array(sorted(d[base.TIME].dt.floor('D').unique()))
 for cut in [days[int(len(days)*q)] for q in (.4,.6,.8)]:
  vi=(d[base.TIME]>=cut)&(d[base.TIME]<cut+pd.Timedelta(days=6));ti=(d[base.TIME]<cut)&(d[base.END]<cut);tr=d[ti&d.side_name.eq('short')];va=d[vi&d.side_name.eq('short')]
  if len(va):*_,score=short.fit_decomp(tr,va,feat,2.);out[va.index.to_numpy()]=score
 d['raw_score']=np.where(d.side_name.eq('short'),out,d.robust_decomposed);return d[np.isfinite(d.raw_score)].copy()
def mapped(history,evaluate,shrink,min_support):
 out=np.full(len(evaluate),np.nan)
 for day in sorted(evaluate[base.TIME].dt.floor('D').unique()):
  ix=evaluate[base.TIME].dt.floor('D').eq(day).to_numpy();h=history[(history[base.TIME]<day)&(history[base.END]<day)&(history[base.TIME]>=day-pd.Timedelta(days=21))]
  if len(h)<min_support or h.raw_score.nunique()<2:out[ix]=evaluate.loc[ix,'raw_score']
  else:
   iso=IsotonicRegression(out_of_bounds='clip').fit(h.raw_score,h[base.Y]);basep=iso.predict(evaluate.loc[ix,'raw_score']);
   if shrink==0:out[ix]=basep;continue
   hp=iso.predict(h.raw_score);res=h[base.Y].to_numpy()-hp
   vals=[]
   for side in evaluate.loc[ix,'side_name']:
    z=res[h.side_name.eq(side).to_numpy()]
    vals.append(float(z.mean()) if len(z)>=min_support else 0.)
   out[ix]=basep+shrink*np.asarray(vals)
 return out
def top(x,col,f):return base.order(x,col,f)
def metric(x,col,label):
 r=[]
 for f in (.01,.05,.1,.2):
  q=top(x,col,f);b=bound(x,col,f);b.update({'map':label,'net_bps':float(q[base.Y].mean()*1e4),'gross_bps':float(q.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(q.execution_cost_return.mean()*1e4),'bias_bps':float((q[col]-q[base.Y]).mean()*1e4),'mae_bps':float(np.abs(q[col]-q[base.Y]).mean()*1e4)});r.append(b)
 return r
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x=load(a);march=march_scores(x);frozen=pd.read_parquet(a.winner/'april_confirmation_predictions.parquet');frozen['__ts__']=pd.to_datetime(frozen['__ts__'],utc=True);april=frozen.copy();april[base.TIME]=pd.to_datetime(april[base.TIME],utc=True);april[base.END]=pd.to_datetime(april[base.END],utc=True)
 # March causal development chooses only overlay strength/support, never a map
 # evaluated on its fit observations.
 choice=[]
 for shrink,support in GRID:
  z=march.copy();z['score']=mapped(march,march,shrink,support);q=top(z,'score',.1);choice.append({'shrink':shrink,'min_support':support,'march_causal_development_top10_net_bps':float(q[base.Y].mean()*1e4)})
 choice=pd.DataFrame(choice).sort_values(['march_causal_development_top10_net_bps','shrink','min_support'],ascending=[False,True,True],kind='mergesort');win=choice.iloc[0].to_dict()
 april['raw']=april.raw_score;frozen_map=IsotonicRegression(out_of_bounds='clip').fit(march.raw_score,march[base.Y]).predict(april.raw_score);april['frozen_march_isotonic']=frozen_map;april['anchor_21d']=mapped(march,april,0.,int(win['min_support']));april['anchor_side_shrink']=mapped(pd.concat([march,april],ignore_index=True),april,float(win['shrink']),int(win['min_support']))
 # Controls have no refit: base/residual/robust scores from exact IDs.
 ctrl=x[x.candidate_month.eq('2025-04')][list(base.ID)+['score_base_alpha','score_residual_expected_ev','direct_q25_return','execution_net_ev_12h','execution_gross_ev_12h','execution_cost_return']].copy();ctrl['__ts__']=pd.to_datetime(ctrl['__ts__'],utc=True);april=april.merge(ctrl,on=list(base.ID)+['execution_net_ev_12h'],validate='one_to_one')
 rows=[];latest=[];side=[];asset=[]
 for label,col in [('raw','raw'),('frozen_march_isotonic','frozen_march_isotonic'),('anchor_21d','anchor_21d'),('anchor_side_shrink','anchor_side_shrink')]:
  rows+=metric(april,col,label)
  for f in (.01,.05,.1,.2):
   q=top(april,col,f);week=april.__ts__.max().floor('D')-pd.Timedelta(days=6);w=top(april[april.__ts__>=week],col,f);latest.append({'map':label,'top_fraction':f,'latest_week_net_bps':float(w.execution_net_ev_12h.mean()*1e4),'latest_week_rows':len(w)})
   if f==.1:
    for s,z in q.groupby('side_name'):side.append({'map':label,'side_name':s,'rows':len(z),'share':len(z)/len(q),'net_bps':float(z.execution_net_ev_12h.mean()*1e4)})
    for s,z in q.groupby('__symbol__'):asset.append({'map':label,'__symbol__':s,'rows':len(z),'share':len(z)/len(q),'net_bps':float(z.execution_net_ev_12h.mean()*1e4)})
 controls=[]
 for col in ['score_base_alpha','score_residual_expected_ev','direct_q25_return']:
  for f in (.01,.05,.1,.2):controls.append({'control':col,'top_fraction':f,'net_bps':float(top(april,col,f).execution_net_ev_12h.mean()*1e4)})
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));march.to_parquet(st/'march_chronological_oof_raw_scores.parquet',index=False);april.to_parquet(st/'april_daily_maps.parquet',index=False);choice.to_csv(st/'march_causal_overlay_selection.csv',index=False);pd.DataFrame(rows).to_csv(st/'mapped_metrics.csv',index=False);pd.DataFrame(latest).to_csv(st/'latest_week.csv',index=False);pd.DataFrame(side).to_csv(st/'side_top10.csv',index=False);pd.DataFrame(asset).to_csv(st/'asset_top10.csv',index=False);pd.DataFrame(controls).to_csv(st/'controls_identical_ids.csv',index=False);outs={p.name:base.hs(p) for p in st.iterdir() if p.is_file()};man={'schema':'short_winner_causal_recent_ev_mapping_v1','status':'RESEARCH_ONLY_NO_REPLAY','contract':{'ranker':'frozen short winner, no April refit','march':'chronological OOF development only; maps never evaluated on their own fit pairs','april':'daily snapshot uses only prior label_end < snapshot, including prior April labels','selection':'pooled global topK no quota','actions':'excluded'},'frozen_overlay':win,'march_rows':len(march),'april_rows':len(april),'inputs':{'winner_manifest':base.hs(a.winner/'manifest.json'),'source':base.hs(a.source),'runner':base.hs(Path(__file__))},'outputs_sha256':outs};base.wj(st/'manifest.json',man);(st/'manifest.sha256').write_text(base.hs(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=base.SRC);p.add_argument('--peak',type=Path,default=base.PEAK);p.add_argument('--slope',type=Path,default=base.SLOPE);p.add_argument('--mae',type=Path,default=MAE);p.add_argument('--winner',type=Path,default=WIN);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,default=str))
