#!/usr/bin/env python3
"""Fit continuous R2 robust-clear target on a frozen R3 feature contract."""
from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
T=pd.Timestamp('2024-03-01',tz='UTC');C=pd.Timestamp('2024-05-01',tz='UTC')
P=dict(n_estimators=300,learning_rate=.04,num_leaves=63,min_child_samples=400,subsample=.8,colsample_bytree=.8,reg_lambda=12.,random_state=20260802,n_jobs=1,verbosity=-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--r3-contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 cols=json.loads((a.r3_contract/'manifest.json').read_text())['selected_features'];files=[a.matrix/x['path'] for x in json.loads((a.matrix/'manifest.json').read_text())['parts']]
 def load(start,end):
  xs=[]
  for f in files:
   x=pd.read_parquet(f,columns=['available','robust_clear_soft_b25_t50','t4_tp6_sl4_net_bps',*cols]);av=pd.to_datetime(x.available,utc=True);xs.append(x.loc[(av>=start)&(av<end)])
  return pd.concat(xs,ignore_index=True)
 tr=load(pd.Timestamp('2020-01-01',tz='UTC'),T);m=lgb.LGBMRegressor(objective='huber',alpha=.9,**P).fit(tr[cols].to_numpy('float32'),tr.robust_clear_soft_b25_t50);del tr;gc.collect();cal=load(T,C);raw=m.predict(cal[cols].to_numpy('float32'));edges=np.unique(np.quantile(raw,np.linspace(0,1,11)));bins=np.clip(np.digitize(raw,edges[1:-1],right=True),0,9);net=cal.t4_tp6_sl4_net_bps.to_numpy();means=np.array([net[bins==i].mean() if (bins==i).any() else net.mean() for i in range(10)]);m.booster_.save_model(str(a.out/'model.txt'));(a.out/'fit_state.json').write_text(json.dumps({'target':'r2','selected_features':cols,'edges':edges.tolist(),'means':means.tolist(),'source_matrix':str(a.matrix)},indent=2))
if __name__=='__main__':main()
