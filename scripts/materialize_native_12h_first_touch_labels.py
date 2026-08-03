#!/usr/bin/env python3
"""Materialise native 12h labels from parity-proven first-touch recipe.

Input is exclusively the full-base exact-1m native path artifact.  The 24h
recipe is preserved, except its 15m time budget is converted to equivalent
minutes and each path is bounded at decision+12h.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_label_first_touch_capture_proxy import _first_touch_capture_outcome
from scripts.run_label_widestop_capture_proxy import CaptureArm

PATHS=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_paths_20260729_v2'
OUT=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1'
LABELS=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
IDENTITY=('candidate_id','side_name','__symbol__','__ts__','__decision_ts__')
GEOM=('__barrier_pct__','__archetype_policy_tp_r__','__archetype_policy_sl_r__','__archetype_policy_trail_r__','__archetype_policy_max_bars_to_mfe__','__archetype_policy_max_barrier__')
def write(path:Path,value:dict[str,Any])->None:path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')
def decode_paths(values:pd.Series)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
 parsed=[json.loads(x) for x in values]
 names=('open','high','low','close')
 arrays=tuple(np.asarray([p[k] for p in parsed],dtype=np.float32) for k in names)
 if any(a.ndim!=2 or a.shape[1]!=720 for a in arrays):raise ValueError('native path is not a complete 720x1m OHLC path')
 return arrays
def prepare_geometry(out:Path,labels_root:Path)->None:
 """Freeze only archived recipe geometry, keyed by accepted candidate id."""
 path=out/'geometry_index.parquet'
 if path.exists():return
 files=[labels_root/f'train_global_{side}_5_2025_{month:02d}.parquet' for month in (1,2,3,4) for side in ('long','short')]
 if not all(p.exists() for p in files):raise FileNotFoundError('missing archived native label shard')
 cols=['candidate_id',*GEOM[1:]]
 geometry=pd.concat([pd.read_parquet(p,columns=cols) for p in files],ignore_index=True)
 if geometry.candidate_id.duplicated().any():raise ValueError('archived geometry candidate ids are not unique')
 out.mkdir(parents=True,exist_ok=True);geometry.to_parquet(path,index=False,compression='zstd')
def label_shard(paths_root:Path,out:Path,shard:str)->None:
 src=paths_root/'shards'/f'{shard}_paths.parquet'; source_manifest=paths_root/'shards'/f'{shard}_manifest.json';dst=out/'shards'/f'{shard}_labels.parquet'; done=out/'shards'/f'{shard}_manifest.json'
 if dst.exists() and done.exists():return
 if not src.exists() or not source_manifest.exists():raise FileNotFoundError(f'{shard}: incomplete path source')
 geometry_path=out/'geometry_index.parquet'
 if not geometry_path.exists():raise FileNotFoundError('run prepare first to freeze archived geometry')
 rows=pd.read_parquet(src).merge(pd.read_parquet(geometry_path),on='candidate_id',how='left',validate='one_to_one')
 if rows[list(GEOM[1:])].isna().any().any():raise ValueError(f'{shard}: missing archived policy geometry')
 paths=decode_paths(rows.native_future_ohlc_path); pieces=[]
 group_cols=('side_name',*GEOM[1:])
 for values,group in rows.groupby(list(group_cols),sort=True,dropna=False):
  pos=group.index.to_numpy(dtype=int); side,*arm_values=values;tp,sl,trail,maxbars_15m,maxbarrier=(float(x) for x in arm_values)
  # The original ``max_bars_to_mfe`` is a 15m duration.  Preserve its time
  # meaning under exact 1m replay; only the overall resolution changes to 12h.
  arm=CaptureArm(name='native_12h_archived_geometry',tp_r=tp,sl_r=sl,trail_r=trail,max_bars_to_mfe=maxbars_15m*15.0,max_barrier=maxbarrier)
  capture=_first_touch_capture_outcome(group,tuple(a[pos] for a in paths),arm,side_name=str(side),outcome_mode='trailing_profit',round_trip_cost=0.01,target_mode='path_ordered',executable_cost_floor=0.01)
  keep=group.loc[:,list(IDENTITY)+list(GEOM)].copy()
  keep['__native_12h_first_touch_target_soft__']=capture.target_soft.to_numpy(dtype=np.float32)
  for old,new in (('capture_net','__native_12h_first_touch_capture_net__'),('capture_hit','__native_12h_first_touch_hit__'),('capture_stop','__native_12h_first_touch_stop__'),('capture_timeout','__native_12h_first_touch_timeout__'),('capture_eligible','__native_12h_first_touch_eligible__'),('capture_valid_path','__native_12h_first_touch_valid_path__'),('first_touch_bar','__native_12h_first_touch_bar__'),('trailing_activated','__native_12h_trailing_activated__'),('trailing_activation_bar','__native_12h_trailing_activation_bar__')):
   keep[new]=pd.to_numeric(capture[old],errors='coerce').to_numpy(dtype=np.float32)
  keep['__native_12h_resolution_ts__']=pd.to_datetime(keep.__decision_ts__,utc=True)+pd.Timedelta(hours=12)
  pieces.append(keep)
 final=pd.concat(pieces,ignore_index=True).sort_values('candidate_id',kind='stable')
 out.mkdir(parents=True,exist_ok=True);(out/'shards').mkdir(exist_ok=True);tmp=dst.with_suffix('.partial');pq.write_table(pa.Table.from_pandas(final,preserve_index=False),tmp,compression='zstd');os.replace(tmp,dst)
 write(done,{'schema':'native_first_touch_12h_label_shard_v1','status':'COMPLETE','shard':shard,'rows':len(final),'source_path_shard':str(src),'source_path_manifest':str(source_manifest),'decision_to_resolution_hours':12,'path_cadence_minutes':1,'path_bars':720,'recipe':{'outcome_mode':'trailing_profit','round_trip_cost':0.01,'target_mode':'path_ordered','max_bars_to_mfe_translation':'archived_15m_bars * 15 -> exact_1m bars'},'outputs':['__native_12h_first_touch_target_soft__','__native_12h_first_touch_capture_net__','__native_12h_first_touch_hit__','__native_12h_first_touch_stop__','__native_12h_first_touch_timeout__']})
def finalize(paths_root:Path,out:Path)->None:
 index=pd.read_parquet(paths_root/'shard_index.parquet');missing=[];ids=[];rows=0;invalid=[]
 for record in index.itertuples(index=False):
  shard=str(record.shard);data=out/'shards'/f'{shard}_labels.parquet';manifest=out/'shards'/f'{shard}_manifest.json'
  if not (data.exists() and manifest.exists()):missing.append(shard);continue
  part=pd.read_parquet(data,columns=['candidate_id','__decision_ts__','__native_12h_resolution_ts__','__native_12h_first_touch_target_soft__'])
  if len(part)!=int(record.rows):invalid.append(shard);continue
  delta=(pd.to_datetime(part.__native_12h_resolution_ts__,utc=True)-pd.to_datetime(part.__decision_ts__,utc=True)).dt.total_seconds()
  target=pd.to_numeric(part.__native_12h_first_touch_target_soft__,errors='coerce')
  if not (delta.eq(12*3600).all() and target.notna().all() and target.between(0.0,1.0).all()):invalid.append(shard);continue
  ids.append(part.candidate_id);rows+=len(part)
 candidate=pd.read_parquet(paths_root/'candidate_inputs.parquet',columns=['candidate_id']).candidate_id
 joined=pd.concat(ids,ignore_index=True) if ids else pd.Series(dtype=object)
 identity_ok=(len(joined)==len(candidate) and not joined.duplicated().any() and set(joined)==set(candidate))
 if missing or invalid or not identity_ok:raise RuntimeError(f'Cannot finalize: missing={len(missing)}, invalid={len(invalid)}, identity_ok={identity_ok}')
 write(out/'manifest.json',{'schema':'native_first_touch_full_12h_labels_v1','status':'COMPLETE','rows':rows,'shards':len(index),'identity_contract':{'source_exact_path_rows':len(candidate),'label_rows':len(joined),'unique_candidate_ids':int(joined.nunique()),'no_overlap':True,'no_missing_or_extra':True},'label_contract':{'target':'__native_12h_first_touch_target_soft__','finite_and_bounded_0_1':True,'resolution':'decision+12h','all_resolution_deltas_seconds':12*3600},'source_paths':str(paths_root),'source_path_completion':str(paths_root/'completion.json'),'strict_native_only':True,'recipe_parity':'24h recipe passed deterministic historical replay audit; max-bars time budget translated from 15m to 1m'})
def run_pending(paths_root:Path,out:Path,limit:int|None)->None:
 index=pd.read_parquet(paths_root/'shard_index.parquet')
 pending=[str(s) for s in index.shard if not ((out/'shards'/f'{s}_labels.parquet').exists() and (out/'shards'/f'{s}_manifest.json').exists())]
 selected=pending if limit is None else pending[:limit]
 for shard in selected:label_shard(paths_root,out,shard)
 print(json.dumps({'requested_limit':limit,'completed_now':len(selected),'remaining_after':len(pending)-len(selected),'shards_total':len(index)}))
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 for name in ('prepare','run-shard','run-pending','finalize'):
  x=sub.add_parser(name);x.add_argument('--paths-root',type=Path,default=PATHS);x.add_argument('--output-dir',type=Path,default=OUT)
  if name=='run-shard':x.add_argument('--shard',required=True)
  if name=='run-pending':x.add_argument('--limit',type=int,default=None)
  if name=='prepare':x.add_argument('--labels-root',type=Path,default=LABELS)
 a=p.parse_args()
 if a.cmd=='prepare':prepare_geometry(a.output_dir,a.labels_root)
 elif a.cmd=='run-shard':label_shard(a.paths_root,a.output_dir,a.shard)
 elif a.cmd=='run-pending':run_pending(a.paths_root,a.output_dir,a.limit)
 else:finalize(a.paths_root,a.output_dir)
if __name__=='__main__':main()
