#!/usr/bin/env python3
"""Resumable native (non-execution-EV) exact-1m path materialisation.

``prepare`` freezes all accepted base-OOF identities plus archived native
geometry and creates deterministic symbol/month shards. ``run-shard`` reads
only the canonical signed Kraken 1m store. It never reads execution-EV labels
or uses their target/cost fields.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.data_store import canonical_kraken_execution_1m_root
from scripts.materialize_execution_entry_timing_1m_paths import _load_symbol_bars,_window_path,HORIZON_MINUTES,IDENTITY,PATH_COLUMNS

BASE=ROOT/'data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet'
LABELS=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
OUT=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_paths_20260729_v2'
NATIVE=("__barrier_pct__","__tp__","__sl__","__first_touch_round_trip_cost__","__first_touch_target_soft__","__first_touch_capture_net__","__first_touch_effective_tp_abs__","__first_touch_effective_sl_abs__","__first_touch_effective_trail_abs__")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(path:Path,payload:dict[str,Any])->None:path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n')
def native_paths(root:Path)->list[Path]:return [root/f'train_global_{side}_5_2025_{month:02d}.parquet' for month in (2,3,4) for side in ('long','short')]
def prepare(out:Path,base_path:Path,labels_root:Path,shard_rows:int)->None:
 manifest_path=out/'manifest.json'; candidate_path=out/'candidate_inputs.parquet'; index_path=out/'shard_index.parquet'
 if manifest_path.exists():raise FileExistsError(out)
 native_files=native_paths(labels_root)
 if not all(p.exists() for p in native_files):raise FileNotFoundError('missing native label shard')
 if candidate_path.exists():
  joined=pd.read_parquet(candidate_path)
  if len(joined)==0:raise ValueError('empty resumable candidate input')
 else:
  base=pd.read_parquet(base_path,columns=[*IDENTITY,'__decision_ts__'])
  base['__ts__']=pd.to_datetime(base.__ts__,utc=True);base['__decision_ts__']=pd.to_datetime(base.__decision_ts__,utc=True)
  cols=['candidate_id','side_name','__ts__',*NATIVE]
  native=pd.concat([pd.read_parquet(p,columns=cols) for p in native_files],ignore_index=True)
  native['__ts__']=pd.to_datetime(native.__ts__,utc=True);native.side_name=native.side_name.astype(str).str.lower()
  joined=base.merge(native,on='candidate_id',how='left',suffixes=('','__native'),validate='one_to_one')
  if len(joined)!=len(base) or joined[list(NATIVE)].isna().any().any():raise ValueError('accepted base identities do not fully join native geometry')
  if not (joined.side_name.eq(joined.pop('side_name__native'))&joined.__ts__.eq(joined.pop('__ts____native'))).all():raise ValueError('native side/time mismatch')
  if not joined.__decision_ts__.eq(joined.__ts__+pd.Timedelta(hours=1)).all():raise ValueError('base decision contract changed')
  out.mkdir(parents=True);joined=joined.sort_values(['__symbol__','__ts__','candidate_id'],kind='stable').reset_index(drop=True)
  joined.to_parquet(candidate_path,index=False,compression='zstd')
 out.mkdir(parents=True,exist_ok=True);(out/'shards').mkdir(exist_ok=True)
 # Deterministic contiguous symbol/month units keep store reads local. No
 # target/outcome influences membership or shard order.
 units=[g for _,g in joined.groupby([joined.__symbol__.astype(str),joined.__ts__.dt.strftime('%Y-%m')],sort=True,observed=True)]
 bucket=[];count=0;plan=[];sid=0
 def flush():
  nonlocal bucket,count,sid
  if not bucket:return
  frame=pd.concat(bucket,ignore_index=True);name=f'shard_{sid:04d}';path=out/'shards'/f'{name}_input.parquet';frame.to_parquet(path,index=False,compression='zstd')
  plan.append({'shard':name,'rows':len(frame),'symbols':int(frame.__symbol__.nunique()),'months':sorted(frame.__ts__.dt.strftime('%Y-%m').unique().tolist()),'input_sha256':sha(path),'input_path':str(path)});sid+=1;bucket=[];count=0
 for unit in units:
  if bucket and count+len(unit)>shard_rows:flush()
  bucket.append(unit);count+=len(unit)
 flush()
 if not plan:raise ValueError('no deterministic path shards created')
 pd.DataFrame(plan).to_parquet(index_path,index=False)
 write(manifest_path,{'schema':'native_first_touch_full_12h_paths_v2','status':'PREPARED_RESUMABLE_NATIVE_PATHS','rows':len(joined),'identity':list(IDENTITY),'source':{'base_oof':{'path':str(base_path),'sha256':sha(base_path)},'native_label_shards':[{'path':str(p),'sha256':sha(p)} for p in native_files]},'native_geometry_columns':list(NATIVE),'timing':{'decision':'signal+1h','path_minutes':720,'cadence_minutes':1,'resolution':'decision+12h'},'shards':len(plan),'shard_index_sha256':sha(index_path),'path_payload_field':'native_future_ohlc_path','explicitly_not_used':['execution EV labels','execution gross/cost/net','execution policy exits']})
def run_shard(root_dir:Path,shard:str,data_root:Path)->None:
 inp=root_dir/'shards'/f'{shard}_input.parquet';out=root_dir/'shards'/f'{shard}_paths.parquet';man=root_dir/'shards'/f'{shard}_manifest.json';missing=root_dir/'shards'/f'{shard}_missing.json'
 if out.exists() and man.exists():return
 if not inp.exists():raise FileNotFoundError(inp)
 rows=pd.read_parquet(inp);store=canonical_kraken_execution_1m_root(data_root)
 records=[];bad=[];parts={}
 for symbol,g in rows.groupby('__symbol__',sort=True):
  start=pd.to_datetime(g.__decision_ts__,utc=True).min();end=pd.to_datetime(g.__decision_ts__,utc=True).max()+pd.Timedelta(minutes=HORIZON_MINUTES)
  bars,source_parts=_load_symbol_bars(store,str(symbol),start,end)
  for p in source_parts:parts[str(p.relative_to(store))]=sha(p)
  for _,row in g.iterrows():
   path,reason,price=_window_path(bars,pd.Timestamp(row['__decision_ts__']))
   if path is None:bad.append({'candidate_id':row.candidate_id,'reason':reason});continue
   # This is raw canonical OHLC input, not an execution-policy label.  Keep
   # the field native so downstream label code cannot accidentally treat it
   # as an execution-EV payload.
   records.append({**{k:row[k] for k in [*IDENTITY,'__decision_ts__',*NATIVE]},'native_future_ohlc_path':path,'decision_price':np.float32(price)})
 write(missing,{'schema':'native_exact_1m_missing_windows_v1','requested':len(rows),'complete':len(records),'missing':bad})
 if bad:raise ValueError(f'{shard}: {len(bad)} incomplete exact paths; see {missing}')
 tmp=out.with_suffix('.partial');pq.write_table(pa.Table.from_pandas(pd.DataFrame(records),preserve_index=False),tmp,compression='zstd');os.replace(tmp,out)
 write(man,{'schema':'native_first_touch_12h_path_shard_v2','status':'COMPLETE','shard':shard,'input_sha256':sha(inp),'output_sha256':sha(out),'rows':len(records),'store':{'root':str(store),'contract':'canonical_kraken_execution_1m_read_only_v1','parts':parts},'timing':{'first_path_timestamp':'__decision_ts__','cadence_minutes':1,'path_minutes':720,'label_resolution':'decision+12h'},'path_payload_field':'native_future_ohlc_path','native_only':True,'not_execution_ev_proxy':True})
def run_pending(root_dir:Path,data_root:Path,limit:int|None)->None:
 """Run deterministic incomplete shards in index order; safe to re-invoke."""
 index=pd.read_parquet(root_dir/'shard_index.parquet')
 pending=[str(s) for s in index.shard if not ((root_dir/'shards'/f'{s}_paths.parquet').exists() and (root_dir/'shards'/f'{s}_manifest.json').exists())]
 selected=pending if limit is None else pending[:limit]
 for shard in selected:run_shard(root_dir,shard,data_root)
 print(json.dumps({'requested_limit':limit,'completed_now':len(selected),'remaining_after':len(pending)-len(selected),'shards_total':len(index)}))
def finalize(root_dir:Path)->None:
 index=pd.read_parquet(root_dir/'shard_index.parquet');bad=[];rows=0;ids=[];hash_mismatches=[]
 for record in index.itertuples(index=False):
  shard=str(record.shard);out=root_dir/'shards'/f'{shard}_paths.parquet';man=root_dir/'shards'/f'{shard}_manifest.json';missing=root_dir/'shards'/f'{shard}_missing.json'
  if not (out.exists() and man.exists() and missing.exists()):bad.append(shard);continue
  detail=json.loads(missing.read_text())
  if detail.get('missing') or int(detail.get('complete',-1))!=int(record.rows):bad.append(shard);continue
  input_path=root_dir/'shards'/f'{shard}_input.parquet'; shard_manifest=json.loads(man.read_text())
  if sha(input_path)!=str(record.input_sha256) or sha(out)!=str(shard_manifest.get('output_sha256')):hash_mismatches.append(shard)
  ids.append(pd.read_parquet(input_path,columns=['candidate_id']).candidate_id)
  rows+=int(record.rows)
 candidate=pd.read_parquet(root_dir/'candidate_inputs.parquet',columns=['candidate_id']).candidate_id
 combined=pd.concat(ids,ignore_index=True)
 identity_ok=(len(combined)==len(candidate) and not combined.duplicated().any() and set(combined)==set(candidate))
 if bad or hash_mismatches or not identity_ok:raise RuntimeError(f'Cannot finalize: bad={len(bad)}, hash_mismatches={len(hash_mismatches)}, identity_ok={identity_ok}')
 completion={'schema':'native_first_touch_full_12h_paths_completion_v2','status':'COMPLETE','rows':rows,'shards':len(index),'all_windows_complete':True,'identity_contract':{'candidate_inputs_rows':len(candidate),'shard_input_rows':len(combined),'unique_candidate_ids':int(combined.nunique()),'no_overlap':True,'no_missing_or_extra':True},'hash_contract':{'shard_input_hashes_verified':len(index),'shard_output_hashes_verified':len(index),'mismatches':[]},'native_only':True,'not_execution_ev_proxy':True}
 write(root_dir/'completion.json',completion)
 root_manifest=json.loads((root_dir/'manifest.json').read_text());root_manifest.update({'status':'COMPLETE_EXACT_1M_PATHS','completion_manifest':str(root_dir/'completion.json'),'completion':completion});write(root_dir/'manifest.json',root_manifest)
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('prepare');a.add_argument('--output-dir',type=Path,default=OUT);a.add_argument('--base-oof',type=Path,default=BASE);a.add_argument('--labels-root',type=Path,default=LABELS);a.add_argument('--shard-rows',type=int,default=3000)
 b=sub.add_parser('run-shard');b.add_argument('--output-dir',type=Path,default=OUT);b.add_argument('--shard',required=True);b.add_argument('--data-root',type=Path,default=ROOT/'data_perp')
 c=sub.add_parser('run-pending');c.add_argument('--output-dir',type=Path,default=OUT);c.add_argument('--data-root',type=Path,default=ROOT/'data_perp');c.add_argument('--limit',type=int,default=None)
 d=sub.add_parser('finalize');d.add_argument('--output-dir',type=Path,default=OUT)
 x=p.parse_args()
 if x.cmd=='prepare':prepare(x.output_dir,x.base_oof,x.labels_root,x.shard_rows)
 elif x.cmd=='run-shard':run_shard(x.output_dir,x.shard,x.data_root)
 elif x.cmd=='run-pending':run_pending(x.output_dir,x.data_root,x.limit)
 else:finalize(x.output_dir)
if __name__=='__main__':main()
