#!/usr/bin/env python3
"""Fail-closed audit of archived native 24h first-touch labels.

The sample is deterministic and balanced by side/month.  It recomputes the
archived trailing recipe against the canonical 15m replay store, using each
row's archived archetype geometry.  It is deliberately a parity check, not an
execution-EV calculation.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_label_first_touch_capture_proxy import _fetch_policy_paths,_first_touch_capture_outcome
from scripts.run_label_widestop_capture_proxy import CaptureArm

LABELS=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
OUT=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_24h_replay_parity_20260729_v1'
MONTHS=(2,3,4); SIDES=('long','short'); N_PER_CELL=8
COMPARE=(
 ('__first_touch_target_soft__','target_soft',1e-6),
 ('__first_touch_capture_net__','capture_net',1e-6),
 ('__first_touch_hit__','capture_hit',0.0),
 ('__first_touch_stop__','capture_stop',0.0),
 ('__first_touch_timeout__','capture_timeout',0.0),
 ('__first_touch_bar__','first_touch_bar',0.0),
)
GEOM=('__archetype_policy_tp_r__','__archetype_policy_sl_r__','__archetype_policy_trail_r__','__archetype_policy_max_bars_to_mfe__','__archetype_policy_max_barrier__')
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def sample() -> pd.DataFrame:
 cols=['candidate_id','side_name','__symbol__','__ts__','__barrier_pct__',*GEOM,*[x[0] for x in COMPARE]]
 frames=[]
 for month in MONTHS:
  for side in SIDES:
   path=LABELS/f'train_global_{side}_5_2025_{month:02d}.parquet'
   x=pd.read_parquet(path,columns=cols)
   # candidate-id hash gives stable, non-outcome-based selection.
   x=x.assign(__sample_hash__=pd.util.hash_pandas_object(x.candidate_id.astype(str),index=False).astype('uint64')).sort_values('__sample_hash__',kind='stable').head(N_PER_CELL)
   frames.append(x)
 out=pd.concat(frames,ignore_index=True);out['__ts__']=pd.to_datetime(out.__ts__,utc=True)
 return out.sort_values(['side_name','__ts__','candidate_id'],kind='stable').reset_index(drop=True)
def paths_for_side(frame:pd.DataFrame,side:str)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
 _,paths,stats=_fetch_policy_paths(frame,labels_path=LABELS,side=side,data_root=ROOT/'data_perp',market_mode='perps',exchange='krakenfutures',path_len=96,apply_delayed_entry=False,timeframe='1h')
 if stats['finite_path_rows'] != len(frame):raise RuntimeError(f'{side}: incomplete canonical 24h paths: {stats}')
 return paths
def main()->None:
 if OUT.exists():raise FileExistsError(OUT)
 src=sample(); recomputed=[]
 for side,side_frame in src.groupby('side_name',sort=True):
  side_frame=side_frame.reset_index().rename(columns={'index':'__source_index__'})
  paths=paths_for_side(side_frame,str(side))
  for values,group in side_frame.groupby(list(GEOM),sort=True,dropna=False):
   positions=group.index.to_numpy(dtype=int); path_group=tuple(p[positions] for p in paths)
   tp,sl,trail,maxbars,maxbarrier=(float(v) for v in values)
   arm=CaptureArm(name='archived_row_geometry',tp_r=tp,sl_r=sl,trail_r=trail,max_bars_to_mfe=maxbars,max_barrier=maxbarrier)
   got=_first_touch_capture_outcome(group,path_group,arm,side_name=str(side),outcome_mode='trailing_profit',round_trip_cost=0.01,target_mode='path_ordered',executable_cost_floor=0.01)
   got['__source_index__']=group.__source_index__.to_numpy();recomputed.append(got.reset_index(drop=True))
 got=pd.concat(recomputed,ignore_index=True).set_index('__source_index__').loc[np.arange(len(src))].reset_index(drop=True)
 audit=src.copy(); audit['__recomputed_index__']=np.arange(len(audit))
 failures=[]; summaries=[]
 for saved,current,tol in COMPARE:
  a=pd.to_numeric(audit[saved],errors='coerce').to_numpy(float); b=pd.to_numeric(got[current],errors='coerce').to_numpy(float)
  equal=(np.isclose(a,b,rtol=0.0,atol=tol,equal_nan=True))
  audit[f'{current}__replayed']=b;audit[f'{current}__abs_diff']=np.abs(a-b)
  summaries.append({'saved':saved,'replayed':current,'tolerance':tol,'matched':int(equal.sum()),'mismatched':int((~equal).sum()),'max_abs_diff':float(np.nanmax(np.abs(a-b))) if np.isfinite(np.abs(a-b)).any() else 0.0})
  if not equal.all():failures.append(current)
 OUT.mkdir(parents=True);audit.to_parquet(OUT/'audit_rows.parquet',index=False,compression='zstd')
 manifest:dict[str,Any]={'schema':'native_24h_first_touch_replay_parity_v1','status':'PARITY_PASS' if not failures else 'PARITY_FAIL_CLOSED','rows':len(audit),'sample_contract':{'sides':list(SIDES),'months':[f'2025-{m:02d}' for m in MONTHS],'rows_per_side_month':N_PER_CELL,'selection':'lowest deterministic candidate_id hash; independent of outcomes'},'recipe':{'path':'canonical_15m_simple_policy_replay','path_len':96,'horizon_hours':24,'signal_to_decision_hours':1,'delayed_entry':False,'outcome_mode':'trailing_profit','round_trip_cost':0.01,'target_mode':'path_ordered','row_geometry':'archived archetype tp/sl/trail/max-bars/max-barrier'},'source':{'labels_root':str(LABELS),'implementation':str(ROOT/'scripts/run_label_first_touch_capture_proxy.py'),'implementation_sha256':sha(ROOT/'scripts/run_label_first_touch_capture_proxy.py')},'comparison':summaries,'failures':failures,'audit_rows':str(OUT/'audit_rows.parquet'),'audit_rows_sha256':sha(OUT/'audit_rows.parquet')}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))
 if failures:raise SystemExit('PARITY_FAIL_CLOSED: '+','.join(failures))
if __name__=='__main__':main()
