#!/usr/bin/env python3
"""Resume scoring a checkpointed disk-backed R3 ladder in bounded batches."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=10);a=p.parse_args()
 state=json.loads((a.fit/'fit_state.json').read_text());cols=state['selected_features'];edges=np.asarray(state['edges']);means=np.asarray(state['means']);model=lgb.Booster(model_file=str(a.fit/'model.txt'));files=[a.matrix/x['path'] for x in json.loads((a.matrix/'manifest.json').read_text())['parts']];root=a.fit/'prediction_parts';root.mkdir(exist_ok=True);start=pd.Timestamp('2024-05-01',tz='UTC');end=pd.Timestamp('2024-12-01',tz='UTC')
 for f in files[a.start:a.start+a.count]:
  dst=root/f.name
  if dst.exists():continue
  x=pd.read_parquet(f,columns=['candidate_id','__ts__','available','t4_tp6_sl4_gross_bps','t4_tp6_sl4_net_bps',*cols]);availability=pd.to_datetime(x.available,utc=True);x=x.loc[(availability>=start)&(availability<end)]
  raw=model.predict(x[cols].to_numpy('float32'));raw=(raw[:,2]-raw[:,0]) if state.get('target','r3')=='r3' else raw;x['score_bps']=means[np.clip(np.digitize(raw,edges[1:-1],right=True),0,9)];x['raw_prediction']=raw;x.rename(columns={'t4_tp6_sl4_gross_bps':'gross_bps','t4_tp6_sl4_net_bps':'net_bps'})[['candidate_id','__ts__','gross_bps','net_bps','score_bps','raw_prediction']].to_parquet(dst,index=False);print(dst.name,flush=True)
if __name__=='__main__':main()
