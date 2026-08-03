#!/usr/bin/env python3
"""Read-only sealed reporting supplement for causality-safe tail repair v2."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile,math
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v2';RAW=ROOT/'data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet';OUT=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v2_supplement_20260730_v1';ID=['candidate_id','side_name','__symbol__','__ts__'];Y='execution_net_ev_12h';ARMS=('incumbent_direct_q25','tail_weighted_direct','robust_decomposed','residual_x_conversion_interaction')
def hs(p):
 d=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def top(x,s,f):
 n=max(1,math.ceil(len(x)*f));z=x.copy()
 for c in ID:z[c]=z[c].astype(str)
 return z.sort_values([s,*ID],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n]
def tie(x,s,f):
 z=top(x,s,f);cut=z[s].iloc[-1];above=x[x[s]>cut];eq=x[np.isclose(x[s],cut,rtol=0,atol=1e-14)];slots=len(z)-len(above);v=eq[Y].to_numpy(float);p=eq.opp.to_numpy(bool)
 def val(a):return float((above[Y].sum()+a.sum())/len(z)*1e4)
 def prec(a):return float((above.opp.sum()+a.sum())/len(z))
 return {'cutoff_tie_rows':len(eq),'cutoff_tie_fraction':len(eq)/len(z),'tie_slots':slots,'expected_net_bps':float((above[Y].sum()+slots*v.mean())/len(z)*1e4),'best_net_bps':val(np.sort(v)[-slots:]),'worst_net_bps':val(np.sort(v)[:slots]),'expected_precision':float((above.opp.sum()+slots*p.mean())/len(z)),'best_precision':prec(np.sort(p)[-slots:]),'worst_precision':prec(np.sort(p)[:slots])}
def ece(x,s):
 q=pd.qcut(x[s],q=min(10,x[s].nunique()),duplicates='drop');g=x.groupby(q,observed=True);return float(sum(len(a)*abs(a[s].mean()-a[Y].mean()) for _,a in g)/len(x)*1e4)
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 m=json.loads((a.source/'manifest.json').read_text());assert (a.source/'manifest.sha256').read_text().split()[0]==hs(a.source/'manifest.json')
 pr=pd.read_parquet(a.source/'confirmation_predictions.parquet');raw=pd.read_parquet(a.raw,columns=ID+['opportunity_gross_above_cost_0bps']);raw['opp']=raw.opportunity_gross_above_cost_0bps.astype(bool);pr=pr.merge(raw[ID+['opp']],on=ID,how='left',validate='many_to_one');rows=[];gates=[]
 for arm in ARMS:
  for stage,s in [('raw',arm),('causal_map','map_'+arm)]:
   x=pr[pr[s].notna()].copy();
   for f in (.01,.05,.1,.2):
    z=top(x,s,f); latest=z[pd.to_datetime(z.execution_decision_utc,utc=True)>=pd.to_datetime(x.execution_decision_utc,utc=True).max()-pd.Timedelta(days=7)]
    for side,ss in [('pooled',z),*[(k,v) for k,v in z.groupby('side_name')]]:
     for asset,aa in [('all',ss),('top_asset',ss[ss['__symbol__'].eq(ss['__symbol__'].value_counts().index[0])])]:
      rows.append({'arm':arm,'stage':stage,'top_fraction':f,'scope':side,'asset_scope':asset,'rows':len(aa),'coverage':len(aa)/len(x),'net_bps':float(aa[Y].mean()*1e4),'positive_rate':float(aa[Y].gt(0).mean()),'opportunity_precision':float(aa.opp.mean()),'asset_share':len(aa)/len(ss) if len(ss) else np.nan,'latest_rows':len(latest) if side=='pooled' and asset=='all' else np.nan,'latest_net_bps':float(latest[Y].mean()*1e4) if len(latest) and side=='pooled' and asset=='all' else np.nan})
    if f==.1:
     t=tie(x,s,f);cal={'calibration_bias_bps':float((x[s]-x[Y]).mean()*1e4),'calibration_mae_bps':float(np.abs(x[s]-x[Y]).mean()*1e4),'calibration_ece_bps':ece(x,s)};gates.append({'arm':arm,'stage':stage,'top10_net_bps':float(z[Y].mean()*1e4),'latest_net_bps':float(latest[Y].mean()*1e4),'min_side_share':float(min(z.side_name.eq('long').mean(),z.side_name.eq('short').mean())),'tie_fraction':t['cutoff_tie_fraction'],'positive_gate':z[Y].mean()>0,'latest_gate':latest[Y].mean()>0,'side_gate':min(z.side_name.eq('long').mean(),z.side_name.eq('short').mean())>=.1,'tie_gate':t['cutoff_tie_fraction']<=.05,'promotion_gate':False,**t,**cal})
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));paths={'side_asset_metrics':st/'side_asset_metrics.csv','tie_calibration_gates':st/'tie_calibration_gates.csv'};pd.DataFrame(rows).to_csv(paths['side_asset_metrics'],index=False);pd.DataFrame(gates).to_csv(paths['tie_calibration_gates'],index=False)
 man={'schema':'bounded_direct_tail_repair_v2_reporting_supplement_v1','status':'COMPLETED_NONPROMOTION','source_v2_manifest_sha256':hs(a.source/'manifest.json'),'contract':{'read_only':'no fit/map/refit; deterministic tie performance is supplemented with expected/best/worst random-tie allocation','gates':'positive/latest/side/tie all required; any plateau >5% fails'},'outputs':{k:{'path':str(a.output_dir/v.name),'sha256':hs(v)} for k,v in paths.items()}};q=st/'manifest.json';q.write_text(json.dumps(man,indent=2)+'\n');(st/'manifest.sha256').write_text(hs(q)+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SRC);p.add_argument('--raw',type=Path,default=RAW);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
