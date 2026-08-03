#!/usr/bin/env python3
"""Prepare January archival-native candidates for the legal February 12h fold."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.materialize_febapr_native_12h_full_paths import NATIVE,IDENTITY,sha,write
LABELS=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
OUT=ROOT/'data_perp/artifacts/january2025_native_first_touch_full_12h_paths_20260729_v1'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 files=[LABELS/f'train_global_{s}_5_2025_01.parquet' for s in ('long','short')]
 cols=[*IDENTITY,'__decision_ts__',*NATIVE]
 x=pd.concat([pd.read_parquet(p,columns=cols) for p in files],ignore_index=True)
 x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['__decision_ts__']=pd.to_datetime(x.__decision_ts__,utc=True)
 if x.candidate_id.duplicated().any() or not x.__decision_ts__.eq(x.__ts__+pd.Timedelta(hours=1)).all():raise ValueError('January native identity/timing contract fails')
 OUT.mkdir(parents=True);(OUT/'shards').mkdir();x=x.sort_values(['__symbol__','__ts__','candidate_id'],kind='stable').reset_index(drop=True);x.to_parquet(OUT/'candidate_inputs.parquet',index=False,compression='zstd')
 plan=[];bucket=[];count=0;sid=0
 def flush():
  nonlocal bucket,count,sid
  if not bucket:return
  frame=pd.concat(bucket,ignore_index=True);name=f'shard_{sid:04d}';p=OUT/'shards'/f'{name}_input.parquet';frame.to_parquet(p,index=False,compression='zstd');plan.append({'shard':name,'rows':len(frame),'symbols':int(frame.__symbol__.nunique()),'months':['2025-01'],'input_sha256':sha(p),'input_path':str(p)});bucket=[];count=0;sid+=1
 for _,unit in x.groupby([x.__symbol__.astype(str),x.__ts__.dt.strftime('%Y-%m')],sort=True,observed=True):
  if bucket and count+len(unit)>3000:flush()
  bucket.append(unit);count+=len(unit)
 flush();pd.DataFrame(plan).to_parquet(OUT/'shard_index.parquet',index=False)
 write(OUT/'manifest.json',{'schema':'january2025_native_first_touch_full_12h_paths_v1','status':'PREPARED_RESUMABLE_NATIVE_PATHS','rows':len(x),'shards':len(plan),'timing':{'decision':'signal+1h','path_minutes':720,'resolution':'decision+12h'},'source_native_ledger':[{'path':str(p),'sha256':sha(p)} for p in files],'explicitly_not_used':['execution EV labels','execution policy exits']})
if __name__=='__main__':main()
