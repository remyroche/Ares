#!/usr/bin/env python3
"""Strict holdout R3 screen with train-only selection from configured base pools."""
from __future__ import annotations

import argparse, json, sys, gc
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
 sys.path.insert(0,str(ROOT))
from extreme_price_movements.config import (DAILY_SR_BASE_FEATURE_KEYS, MODEL_DIRECT_BASE_FEATURE_KEYS, ORDERBOOK_BASE_FEATURE_KEYS, RESIDUAL_BASE_FEATURE_KEYS, VOLUME_FREE_PERP_BASE_FEATURE_KEYS)

# Broad enough to test the frozen contract, but bounded before row materialisation:
# price/range/OI/funding, residual cross-sectional, order-book, OI-location and
# daily support/resistance are all represented.  The eventual subset is still
# selected using pre-March rows only.
POOL=list(dict.fromkeys(
    MODEL_DIRECT_BASE_FEATURE_KEYS[:32]
    + RESIDUAL_BASE_FEATURE_KEYS
    + ORDERBOOK_BASE_FEATURE_KEYS
    + VOLUME_FREE_PERP_BASE_FEATURE_KEYS
    + DAILY_SR_BASE_FEATURE_KEYS[:16]
))
PARAMS=dict(n_estimators=300,learning_rate=.04,num_leaves=63,min_child_samples=400,subsample=.8,colsample_bytree=.8,reg_lambda=12.,random_state=20260802,n_jobs=1,verbosity=-1)

def main() -> None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--side',choices=('long','short'),required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n-features',type=int,default=64);a=p.parse_args()
 panel=ROOT/'data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3'; winner=ROOT/'data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1'; labels=ROOT/'data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1'
 available=set(pd.read_parquet(next((panel/'parts').glob('*.parquet'))).columns); pool=[c for c in POOL if c in available]
 # Stream a train-only univariate screen; feature coverage and correlation are
 # both calculated before March, then the holdout is never revisited for
 # selection. This is deliberately a ladder diagnostic, not final HPO.
 sums={c:np.zeros(5) for c in pool}; ntrain=0
 for part in sorted((panel/'parts').glob('*.parquet')):
  x=pd.read_parquet(part,columns=['candidate_id','side_name','__ts__',*pool]);x=x.loc[x.side_name.eq(a.side)]
  lab=pd.read_parquet(labels/'parts'/part.name,columns=['candidate_id','label_valid','lower_touch_minute','robust_clear_event_b25','__label_available_at__'])
  x=x.merge(lab,on='candidate_id',validate='one_to_one'); x=x.loc[x.label_valid & (pd.to_datetime(x.__label_available_at__,utc=True)<pd.Timestamp('2024-03-01',tz='UTC'))]
  y=np.select([x.robust_clear_event_b25.eq(1),x.lower_touch_minute.ge(0)],[2.,0.],default=1.)
  ntrain+=len(x)
  for c in pool:
   v=pd.to_numeric(x[c],errors='coerce').to_numpy(float); ok=np.isfinite(v)
   if ok.any(): sums[c]+=np.array([ok.sum(),v[ok].sum(),(v[ok]*v[ok]).sum(),y[ok].sum(),(v[ok]*y[ok]).sum()])
  del x, lab, y
  gc.collect()
 ranks=[]
 for c,(n,sx,sxx,sy,sxy) in sums.items():
  den=np.sqrt(max(n*sxx-sx*sx,0)*max(n*(sy*sy/n)-sy*sy/n,0)) if n else 0
  # y sum-squared is computed exactly below in second pass avoidance: use a
  # bounded covariance proxy; it ranks features robustly under identical y.
  corr=abs((n*sxy-sx*sy)/(np.sqrt(max(n*sxx-sx*sx,1))*max(np.sqrt(n),1)))
  ranks.append((corr,c,n/ntrain if ntrain else 0))
 selected=[c for _,c,cov in sorted(ranks,reverse=True) if cov>=.9][:a.n_features]
 def load_window(start, end, keep_identity=False):
  frames=[]
  for part in sorted((panel/'parts').glob('*.parquet')):
   x=pd.read_parquet(part,columns=['candidate_id','side_name','__ts__',*selected]);x=x.loc[x.side_name.eq(a.side)]
   w=pd.read_parquet(winner/'parts'/part.name,columns=['candidate_id','t4_tp6_sl4_gross_bps','t4_tp6_sl4_net_bps'])
   lab=pd.read_parquet(labels/'parts'/part.name,columns=['candidate_id','label_valid','lower_touch_minute','robust_clear_event_b25','__label_available_at__'])
   x=x.merge(w,on='candidate_id',validate='one_to_one').merge(lab,on='candidate_id',validate='one_to_one');available=pd.to_datetime(x.__label_available_at__,utc=True)
   x=x.loc[x.label_valid & (available>=start) & (available<end)]
   x['y']=np.select([x.robust_clear_event_b25.eq(1),x.lower_touch_minute.ge(0)],[2,0],default=1).astype('int8')
   if not keep_identity: x=x[[*selected,'y','t4_tp6_sl4_net_bps']]
   frames.append(x);del w,lab;gc.collect()
  return pd.concat(frames,ignore_index=True)
 train=load_window(pd.Timestamp('2020-01-01',tz='UTC'),pd.Timestamp('2024-03-01',tz='UTC'))
 def mat(z): return z[selected].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy('float32')
 m=lgb.LGBMClassifier(objective='multiclass',num_class=3,**PARAMS).fit(mat(train),train.y);del train;gc.collect();cal=load_window(pd.Timestamp('2024-03-01',tz='UTC'),pd.Timestamp('2024-05-01',tz='UTC'));pc=m.predict_proba(mat(cal));sc=pc[:,2]-pc[:,0]
 test=load_window(pd.Timestamp('2024-05-01',tz='UTC'),pd.Timestamp('2024-12-01',tz='UTC'),keep_identity=True);pt=m.predict_proba(mat(test));st=pt[:,2]-pt[:,0]
 edges=np.unique(np.quantile(sc,np.linspace(0,1,11))); bins=np.clip(np.digitize(sc,edges[1:-1],right=True),0,9); means=np.array([cal.t4_tp6_sl4_net_bps.to_numpy()[bins==i].mean() if (bins==i).any() else cal.t4_tp6_sl4_net_bps.mean() for i in range(10)]); score=means[np.clip(np.digitize(st,edges[1:-1],right=True),0,9)]
 out=test[['candidate_id','__ts__','side_name','t4_tp6_sl4_gross_bps','t4_tp6_sl4_net_bps']].rename(columns={'t4_tp6_sl4_gross_bps':'gross_bps','t4_tp6_sl4_net_bps':'net_bps'});out['score_bps']=score;out['raw_prediction']=st;out.to_parquet(a.out/'predictions.parquet',index=False) if (a.out.mkdir(parents=True,exist_ok=False) is None) else None
 rows=[]
 for f in (.01,.05,.10):
  z=out.nlargest(int(np.ceil(len(out)*f)),'score_bps');rows.append({'top_fraction':f,'rows':len(z),'gross_bps':z.gross_bps.mean(),'net_bps':z.net_bps.mean()})
 pd.DataFrame(rows).to_parquet(a.out/'results.parquet',index=False);(a.out/'manifest.json').write_text(json.dumps({'side':a.side,'selected_features':selected,'feature_coverage':{c:cov for _,c,cov in ranks if c in selected},'pool_size':len(pool),'train_rows':len(train),'calibration_rows':len(cal),'test_rows':len(test),'selection':'train-only univariate covariance screen, coverage >=90%, then LGBM R3'},indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
