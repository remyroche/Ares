#!/usr/bin/env python3
import argparse,hashlib,json,math,os,tempfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2'
def h(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def select(x,f):
 n=max(1,math.ceil(len(x)*f));return x.sort_values(['raw_score','candidate_id','__ts__','__symbol__','side_name'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n].copy()
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x=pd.read_parquet(a.source/'april_confirmation_predictions.parquet');x['__ts__']=pd.to_datetime(x['__ts__'],utc=True);rng=np.random.default_rng(20260730);rows=[];side=[]
 for f in (.01,.1):
  q=select(x,f);q['day']=q.__ts__.dt.floor('D');days=np.array(sorted(q.day.unique()));means=[]
  for _ in range(2000):
   take=rng.choice(days,size=len(days),replace=True);z=pd.concat([q[q.day.eq(d)] for d in take],ignore_index=True);means.append(float(z.execution_net_ev_12h.mean()*1e4))
  rows.append({'score_kind':'raw','top_fraction':f,'selected_rows':len(q),'utc_days':len(days),'day_start_utc':str(days.min()),'day_end_utc':str(days.max()),'net_bps':float(q.execution_net_ev_12h.mean()*1e4),'ci95_low_bps':float(np.quantile(means,.025)),'ci95_high_bps':float(np.quantile(means,.975)),'bootstrap_replicates':2000,'contract':'fixed frozen April book and raw score; UTC-day block resampling only; no reselection'})
  for s,z in q.groupby('side_name'):side.append({'top_fraction':f,'side_name':s,'rows':len(z),'share':len(z)/len(q),'net_bps':float(z.execution_net_ev_12h.mean()*1e4)})
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));pd.DataFrame(rows).to_csv(st/'utc_day_block_intervals.csv',index=False);pd.DataFrame(side).to_csv(st/'top_side_attribution.csv',index=False);m={'schema':'bounded_short_conditional_payoff_day_block_v1','status':'SEALED_REPORTING_ONLY','source_manifest_sha256':h(a.source/'manifest.json'),'source_predictions_sha256':h(a.source/'april_confirmation_predictions.parquet'),'outputs_sha256':{p.name:h(p) for p in st.iterdir() if p.is_file()},'runner_sha256':h(Path(__file__))};(st/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SRC);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
