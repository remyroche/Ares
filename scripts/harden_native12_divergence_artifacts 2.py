#!/usr/bin/env python3
"""Publish immutable v2 provenance manifests for frozen native12 divergence reports."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'data_perp/artifacts/febapr2025_native12h_matched_score_divergence_20260729_v1/identical_rows.parquet'
EV=ROOT/'data_perp/artifacts/febapr2025_native12h_execution_ev_divergence_20260729_v1/joined_scores_execution_ev.parquet'
OUT=ROOT/'data_perp/artifacts/febapr2025_native12h_divergence_provenance_20260729_v2'
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def identity_hash(ids:pd.Series)->str:
 return hashlib.sha256(pd.util.hash_pandas_object(ids.astype(str).sort_values(kind='stable'),index=False).values.tobytes()).hexdigest()
def publish(out:Path=OUT,native:Path=NATIVE,ev:Path=EV)->dict:
 if out.exists():raise FileExistsError(out)
 n=pd.read_parquet(native,columns=['candidate_id']);e=pd.read_parquet(ev,columns=['candidate_id','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h'])
 if len(n)!=509868 or len(e)!=len(n) or n.candidate_id.duplicated().any() or e.candidate_id.duplicated().any() or set(n.candidate_id)!=set(e.candidate_id):raise ValueError('exact identity contract fails')
 if not np.allclose(e.execution_gross_ev_12h-e.execution_cost_return,e.execution_net_ev_12h,atol=1e-12,rtol=0.0,equal_nan=False):raise ValueError('gross-cost-net accounting assertion fails')
 value={'schema':'native12_divergence_provenance_v2','status':'IMMUTABLE_FROZEN_OOF_HELDOUT_EV','rows':len(e),'identity':{'unique_rows':int(e.candidate_id.nunique()),'candidate_id_sha256':identity_hash(e.candidate_id),'matches_native_score_rows':True},'accounting':{'assertion':'execution_gross_ev_12h - execution_cost_return == execution_net_ev_12h','passed':True,'tolerance':1e-12},'sources':{'native_identical_scores':{'path':str(native),'sha256':sha(native)},'heldout_execution_join':{'path':str(ev),'sha256':sha(ev)}}}
 out.mkdir(parents=True);tmp=out/'manifest.json.tmp'
 tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'manifest.json');return value
if __name__=='__main__':print(json.dumps(publish(),indent=2))
