#!/usr/bin/env python3
"""Strict pre-2026 fixed convex mixture of the three final-v3 GAM experts."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data_perp/artifacts'
SOURCE=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3'
OUT=ART/'final_v3_gam_convex_mixture_ablation_20260730_v1'
EXPERTS=('gam_regime_only','gam_transition_only','gam_combined')
BASE='baseline'; TOP=.10

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def grid():
 for a in (0.,.25,.5,.75,1.):
  for b in (0.,.25,.5,.75,1.):
   c=1-a-b
   if c>=0 and c in (0.,.25,.5,.75,1.):yield (a,b,c)
def top(frame,score):
 n=max(1,int(np.ceil(len(frame)*TOP)))
 return frame.nlargest(n,score,keep='all').sort_values(score,ascending=False,kind='stable').head(n)
def periods(frame):
 x=frame.copy();x['week']=x.__ts__.dt.to_period('W').astype(str);x['month']=x.__ts__.dt.strftime('%Y-%m')
 out=[]
 for unit in ('week','month'):
  v=x.groupby(unit).execution_net_ev_12h.mean()
  out.extend([{'unit':unit,'stat':'q10','value':float(v.quantile(.1))},{'unit':unit,'stat':'q50','value':float(v.quantile(.5))}])
 return out
def audit(frame,name):
 book=top(frame,'mapped_score'); rows=[{'mixture':name,'metric':'aggregate_net_bps','value':float(book.execution_net_ev_12h.mean()*1e4)},{'mixture':name,'metric':'both_side_min_net_bps','value':float(book.groupby('side_name').execution_net_ev_12h.mean().min()*1e4)}]
 rows += [{'mixture':name,'metric':f"{x['unit']}_{x['stat']}_net_bps",'value':x['value']*1e4} for x in periods(book)]
 return rows,book
def main():
 if OUT.exists():raise FileExistsError(OUT)
 hist=pd.read_parquet(SOURCE/'historical_oof_scores.parquet');hist.__ts__=pd.to_datetime(hist.__ts__,utc=True)
 key=['candidate_id','__ts__','__symbol__','side_name','execution_label_end_utc','execution_net_ev_12h']
 h=hist.loc[hist.arm.isin(EXPERTS),key+['arm','raw_score']].pivot(index=key,columns='arm',values='raw_score').reset_index();h.columns.name=None
 f=pd.read_parquet(SOURCE/'frozen_2026_candidate_scores.parquet');f.__ts__=pd.to_datetime(f.__ts__,utc=True)
 f=f.loc[f.arm.isin(EXPERTS),key+['arm','raw_score']].pivot(index=key,columns='arm',values='raw_score').reset_index();f.columns.name=None
 rows=[]; candidates=[]
 for w in grid():
  name='mix_'+'_'.join(str(int(x*100)) for x in w)
  raw=sum(weight*h[arm] for weight,arm in zip(w,EXPERTS)); mapper=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(raw,h.execution_net_ev_12h)
  hh=h.assign(raw_score=raw,mapped_score=mapper.predict(raw)); rr,book=audit(hh,name);rows+=rr
  fr=sum(weight*f[arm] for weight,arm in zip(w,EXPERTS)); ff=f.assign(raw_score=fr,mapped_score=mapper.predict(fr));rr,book2026=audit(ff,name);rows += [{**x,'scope':'untouched_2026'} for x in rr]; candidates.append(book2026.assign(mixture=name))
 metrics=pd.DataFrame(rows); histm=metrics.loc[~metrics.get('scope',pd.Series(index=metrics.index,dtype=object)).eq('untouched_2026')].pivot(index='mixture',columns='metric',values='value').reset_index()
 gates=(histm.aggregate_net_bps>0)&(histm.week_q10_net_bps>=0)&(histm.week_q50_net_bps>=0)&(histm.month_q10_net_bps>=0)&(histm.month_q50_net_bps>=0)&(histm.both_side_min_net_bps>=0)
 histm['oof_promotion_gate_passed']=gates
 diagnostic=histm.sort_values(['aggregate_net_bps','week_q10_net_bps','both_side_min_net_bps'],ascending=False).iloc[0]
 stage=Path(tempfile.mkdtemp(dir=OUT.parent,prefix=f'.{OUT.name}.'))
 try:
  metrics.to_csv(stage/'mixture_oof_and_forward_metrics.csv',index=False);histm.to_csv(stage/'historical_oof_gate_table.csv',index=False);pd.concat(candidates,ignore_index=True).to_parquet(stage/'frozen_2026_mixture_top10_books.parquet',index=False)
  report={'status':'SEALED_NON_PROMOTION_FIXED_CONVEX_GRID','experts':EXPERTS,'weights':'0,.25,.5,.75,1 summing to 1','selection':'historical pre-2026 OOF aggregate+week/month Q10/Q50+both-side gates only','gate_blender':'not run; fixed convex grid only','winner_if_any':None if not gates.any() else histm.loc[gates].sort_values('aggregate_net_bps',ascending=False).mixture.iloc[0],'diagnostic_best':diagnostic.mixture,'all_historical_gates_passed':bool(gates.any()),'model_sample_cadence':'1h','assessment_sample_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','no_2026_tuning':True,'promotion_eligible':False}
  (stage/'report.json').write_text(json.dumps(report,indent=2)+'\n'); files=[p for p in stage.iterdir() if p.is_file()];manifest={**report,'schema':'final_v3_gam_convex_mixture_ablation_v1','inputs':{str(SOURCE/'manifest.json'):sha(SOURCE/'manifest.json'),str(SOURCE/'historical_oof_scores.parquet'):sha(SOURCE/'historical_oof_scores.parquet'),str(SOURCE/'frozen_2026_candidate_scores.parquet'):sha(SOURCE/'frozen_2026_candidate_scores.parquet')},'outputs_sha256':{p.name:sha(p) for p in files}};mp=stage/'manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(f'{sha(mp)}  manifest.json\n');os.replace(stage,OUT)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
