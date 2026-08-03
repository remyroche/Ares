#!/usr/bin/env python3
"""Replay 21-day causal side-local expected-net admission before global rank."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def map_score(history: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    if len(history)<200: return np.full(len(raw),np.nan)
    edges=np.unique(np.quantile(history.raw_prediction,np.linspace(0,1,11)))
    if len(edges)<3:return np.full(len(raw),history.net_bps.mean())
    hb=np.clip(np.digitize(history.raw_prediction,edges[1:-1],right=True),0,9)
    # Mild empirical-Bayes shrinkage prevents sparse bins from getting a
    # false high expected net; prior is side-local trailing mean.
    prior=history.net_bps.mean();count=np.bincount(hb,minlength=10);total=np.bincount(hb,weights=history.net_bps,minlength=10);means=(total+50*prior)/(count+50)
    return means[np.clip(np.digitize(raw,edges[1:-1],right=True),0,9)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--long',type=Path,required=True);p.add_argument('--short',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--window-days',type=int,default=21);p.add_argument('--threshold-bps',type=float,default=50.);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 frames=[]
 for side,path in [('long',a.long),('short',a.short)]:
  x=pd.read_parquet(path);x['side_name']=side;x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x=x.sort_values('__ts__').reset_index(drop=True);x['decision_day']=x.__ts__.dt.floor('D');maps=[]
  for day,idx in x.groupby('decision_day',sort=True).groups.items():
   # Daily refresh avoids an expensive per-hour recomputation while remaining
   # strictly causal: the newest history day is D-2, so every H12 label is
   # resolved before the first decision on D.
   h=x.loc[(x.decision_day>=day-pd.Timedelta(days=a.window_days+1))&(x.decision_day<=day-pd.Timedelta(days=2))]
   maps.append(pd.DataFrame({'idx':list(idx),'causal_expected_net_bps':map_score(h,x.loc[list(idx),'raw_prediction'].to_numpy())}))
  mapped=pd.concat(maps,ignore_index=True).set_index('idx');x['causal_expected_net_bps']=mapped.causal_expected_net_bps.reindex(x.index);x['admitted']=x.causal_expected_net_bps.ge(a.threshold_bps);frames.append(x)
 x=pd.concat(frames,ignore_index=True);x.to_parquet(a.out/'causal_admission_predictions.parquet',index=False)
 rows=[]
 for side,g in x.groupby('side_name'):
  z=g[g.admitted];rows.append({'scope':side,'eligible_rows':len(z),'eligible_fraction':len(z)/len(g),'gross_bps':z.gross_bps.mean(),'net_bps':z.net_bps.mean()})
 admitted=x[x.admitted].dropna(subset=['causal_expected_net_bps'])
 for f in (.01,.05,.10):
  # Percentage is of the original candidate population; rank only admitted
  # rows so top-k stays global, not timestamp-local.
  z=admitted.nlargest(min(len(admitted),int(np.ceil(len(x)*f))),'causal_expected_net_bps');rows.append({'scope':f'global_top_{f:.0%}_after_admission','eligible_rows':len(z),'eligible_fraction':len(z)/len(x),'gross_bps':z.gross_bps.mean() if len(z) else np.nan,'net_bps':z.net_bps.mean() if len(z) else np.nan,'long_rows':int((z.side_name=='long').sum()),'short_rows':int((z.side_name=='short').sum())})
 pd.DataFrame(rows).to_parquet(a.out/'target_mapping_ablation.parquet',index=False);(a.out/'run_manifest.json').write_text(json.dumps({'mapping':'side-local 21-day trailing, label availability decision+12h strictly before score time; 10 bins with 50-row side prior shrinkage','threshold_bps':a.threshold_bps,'global_ranking':'after side-local absolute admission'},indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
