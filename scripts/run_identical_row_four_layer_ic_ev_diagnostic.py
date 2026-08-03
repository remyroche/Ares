#!/usr/bin/env python3
"""Sealed identical-row raw-base → residual → direct → mapped-EV diagnostic."""
from __future__ import annotations

import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet'
RAW=ROOT/'data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet'
OUT=ROOT/'data_perp/artifacts/identical_row_four_layer_ic_ev_diagnostic_20260730_v1'
IDENTITY=('candidate_id','side_name','__symbol__','__ts__')
TOPS=(.01,.05,.10,.20)
LAYERS={
 'raw_base_alpha':('score_base_alpha','unitless_native_alpha'),
 'residual_alpha_expected_ev':('score_residual_expected_ev','return'),
 'direct_ev_q25':('score_direct_q25_bps','bps'),
 'causal_mapped_ev':('mapped_direct_net','return'),
}
LABELS=('execution_net_ev_12h','execution_gross_ev_12h','execution_cost_return','execution_mfe_return_12h','execution_mae_return_12h','execution_exit_reason','opportunity_gross_above_cost_0bps')

def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as h:
  for b in iter(lambda:h.read(1<<20),b''): d.update(b)
 return d.hexdigest()
def writej(p:Path,x:Mapping[str,Any])->None:
 def safe(v):
  if isinstance(v,Mapping): return {str(k):safe(a) for k,a in v.items()}
  if isinstance(v,(list,tuple)): return [safe(a) for a in v]
  if isinstance(v,(np.floating,np.integer)): return v.item()
  if isinstance(v,(Path,pd.Timestamp)): return str(v)
  if isinstance(v,float) and not np.isfinite(v): return None
  return v
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(safe(x),indent=2,sort_keys=True)+'\n');os.replace(t,p)
def top(frame:pd.DataFrame,score:str,f:float)->pd.DataFrame:
 n=max(1,math.ceil(len(frame)*f)); w=frame.copy()
 # candidate ID then remainder makes ties total, while the book stays pooled.
 for c in IDENTITY:w[c]=w[c].astype(str)
 return w.sort_values([score,'candidate_id','__ts__','__symbol__','side_name'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n]
def ric(a:pd.Series,b:pd.Series)->float:
 x=pd.DataFrame({'a':a,'b':b}).apply(pd.to_numeric,errors='coerce').dropna()
 return float(x.a.corr(x.b,method='spearman')) if len(x)>=3 and x.a.nunique()>1 and x.b.nunique()>1 else np.nan
def unit_score(frame:pd.DataFrame,layer:str)->np.ndarray:
 col,unit=LAYERS[layer]; x=frame[col].to_numpy(float); return x/1e4 if unit=='bps' else x
def tail_rows(frame:pd.DataFrame,layer:str)->list[dict[str,Any]]:
 score,unit=LAYERS[layer]; allpos=frame.opportunity_gross_above_cost_0bps.astype(bool).sum(); rows=[]
 for f in TOPS:
  s=top(frame,score,f); net=s.execution_net_ev_12h.astype(float); gross=s.execution_gross_ev_12h.astype(float); cost=s.execution_cost_return.astype(float); opp=s.opportunity_gross_above_cost_0bps.astype(bool); ex=s.execution_exit_reason.astype(str).str.lower()
  pred=unit_score(s,layer); calib=unit!='unitless_native_alpha'
  r={'layer':layer,'score_column':score,'score_unit':unit,'top_fraction':f,'candidate_rows':len(frame),'selected_rows':len(s),'full_rank_ic_net':ric(frame[score],frame.execution_net_ev_12h),'tail_rank_ic_net':ric(s[score],net),'opportunity_precision':float(opp.mean()),'opportunity_recall':float(opp.sum()/allpos) if allpos else np.nan,'mean_gross_bps':float(gross.mean()*1e4),'mean_deployed_cost_bps':float(cost.mean()*1e4),'mean_net_deployed_bps':float(net.mean()*1e4),'mean_net_zero_cost_bps':float(gross.mean()*1e4),'fixed_book_cost_drag_bps':float(cost.mean()*1e4),'positive_net_rate':float(net.gt(0).mean()),'positive_net_magnitude_bps':float(net[net.gt(0)].mean()*1e4) if net.gt(0).any() else np.nan,'adverse_net_loss_bps':float(net[net.le(0)].mean()*1e4) if net.le(0).any() else np.nan,'mean_mfe_bps':float(s.execution_mfe_return_12h.mean()*1e4),'mean_mae_bps':float(s.execution_mae_return_12h.mean()*1e4),'calibration_supported':calib,'tail_prediction_bias_bps':float((pred-net.to_numpy()).mean()*1e4) if calib else np.nan,'tail_prediction_mae_bps':float(np.abs(pred-net.to_numpy()).mean()*1e4) if calib else np.nan,'cutoff_tie_rows':int(np.isclose(frame[score].to_numpy(float),s[score].iloc[-1],rtol=0,atol=1e-14).sum()),'score_distinct_values':int(frame[score].nunique())}
  for name in ('trailing','timeout','full_stop','adverse_exit'):
   m=ex.eq(name);r['exit_'+name+'_rate']=float(m.mean())
  rows.append(r)
 return rows
def attr(frame:pd.DataFrame,layer:str)->list[pd.DataFrame]:
 score,_=LAYERS[layer];out=[]
 for f in TOPS:
  s=top(frame,score,f)
  for kind,col in [('side','side_name'),('asset','__symbol__')]:
   g=s.groupby(col,observed=True).agg(selected_rows=('candidate_id','size'),mean_net_bps=('execution_net_ev_12h',lambda x:float(x.mean()*1e4)),positive_rate=('execution_net_ev_12h',lambda x:float(x.gt(0).mean())),opportunity_precision=('opportunity_gross_above_cost_0bps',lambda x:float(x.astype(bool).mean()))).reset_index().rename(columns={col:'bucket'})
   g['layer']=layer;g['top_fraction']=f;g['attribution_kind']=kind;g['selected_share']=g.selected_rows/len(s);out.append(g)
 return out
def readiness(panel:pd.DataFrame,raw:pd.DataFrame,joined:pd.DataFrame)->pd.DataFrame:
 return pd.DataFrame([
  {'source':'marapr2025','status':'READY_EXACT_FOUR_LAYER','raw_rows':len(raw),'canonical_rows':len(panel),'exact_identity_rows':len(joined),'missing_requirement':None},
  {'source':'mayjul2026','status':'FAIL_CLOSED_NO_CANONICAL_MAPPED_EV_ON_IDENTICAL_ROWS','raw_rows':127777,'canonical_rows':0,'exact_identity_rows':0,'missing_requirement':'causal mapped EV score with exact canonical alpha identity and exact 12h labels'},
 ])
def run(a:argparse.Namespace)->dict[str,Any]:
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 p=pd.read_parquet(a.panel,columns=[*IDENTITY,'candidate_month','base_oof_score','mapped_direct_net','mapped_eligible',*LABELS]);p=p[p.candidate_month.astype(str).isin(['2025-03','2025-04'])]
 r=pd.read_parquet(a.raw); required=set(IDENTITY)|set(LABELS)|{'score_base_alpha','score_residual_expected_ev','score_direct_q25_bps','candidate_month'}; miss=required-set(r)
 if miss:raise ValueError('raw source missing '+str(sorted(miss)))
 j=r.merge(p,on=list(IDENTITY),how='inner',suffixes=('_raw','_mapped'),validate='one_to_one')
 if len(j)!=len(r) or not j.mapped_eligible.all() or j.mapped_direct_net.isna().any():raise RuntimeError('four-layer mapped identity coverage incomplete')
 for col in LABELS:
  if col=='opportunity_gross_above_cost_0bps':
   if not j[col+'_raw'].astype(bool).eq(j[col+'_mapped'].astype(bool)).all():raise RuntimeError('label mismatch '+col)
  elif not np.array_equal(j[col+'_raw'].to_numpy(),j[col+'_mapped'].to_numpy()):raise RuntimeError('label mismatch '+col)
 work=j[[*IDENTITY,'candidate_month_raw','score_base_alpha','score_residual_expected_ev','score_direct_q25_bps','mapped_direct_net',*[x+'_raw' for x in LABELS]]].copy();work=work.rename(columns={'candidate_month_raw':'candidate_month',**{x+'_raw':x for x in LABELS}})
 if work.duplicated(list(IDENTITY)).any() or not np.allclose(work.execution_gross_ev_12h-work.execution_cost_return,work.execution_net_ev_12h,atol=1e-12):raise RuntimeError('identity/economic contract failed')
 tails=pd.DataFrame(sum((tail_rows(work,l) for l in LAYERS),[])); at=pd.concat(sum((attr(work,l) for l in LAYERS),[]),ignore_index=True)
 # Selected identities are frozen once per layer/fraction; zero/deployed costs above use exactly these books.
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));paths={'readiness':st/'readiness.csv','tail_metrics':st/'tail_metrics.csv','attribution':st/'side_asset_attribution.csv','common_rows':st/'common_rows.parquet'}
 readiness(p,r,work).to_csv(paths['readiness'],index=False);tails.to_csv(paths['tail_metrics'],index=False);at.to_csv(paths['attribution'],index=False);work.to_parquet(paths['common_rows'],index=False)
 m={'schema':'identical_row_four_layer_ic_ev_diagnostic_v1','status':'MARAPR_READY_MAYJUL_FAIL_CLOSED_NO_SUBSTITUTION','promotion_eligible':False,'contract':{'population':'exact one-to-one Mar-Apr 2025 canonical alpha intersection, 140682 rows','layers':LAYERS,'selection':'pooled global only, candidate-ID stable ties; side/asset are attribution after selection','cost_sensitivity':'zero-cost gross and deployed-cost net computed on identical selected books without reselection','mayjul':'explicit readiness failure; no non-identical mapped score substituted'},'inputs':{'panel':{'path':str(a.panel),'sha256':sha(a.panel)},'raw':{'path':str(a.raw),'sha256':sha(a.raw)}},'outputs':{k:{'path':str(a.output_dir/v.name),'sha256':sha(v)} for k,v in paths.items()}}
 writej(st/'manifest.json',m);(st/'manifest.sha256').write_text(sha(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return m
def parser():
 q=argparse.ArgumentParser();q.add_argument('--panel',type=Path,default=PANEL);q.add_argument('--raw',type=Path,default=RAW);q.add_argument('--output-dir',type=Path,required=True);return q
if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2,default=str))
