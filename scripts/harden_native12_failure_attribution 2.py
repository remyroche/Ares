#!/usr/bin/env python3
"""Freeze hashes and identity contract for the diagnostic-only attribution artifact."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data_perp/artifacts/febapr2025_native12_execution_ev_failure_attribution_20260729_v1'
OUT=ROOT/'data_perp/artifacts/febapr2025_native12_execution_ev_failure_attribution_20260729_v2'
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 rows=pd.read_parquet(SRC/'joined_frozen_attribution_rows.parquet',columns=['candidate_id'])
 if len(rows)!=509868 or rows.candidate_id.duplicated().any():raise ValueError('identity failure')
 ident=hashlib.sha256(pd.util.hash_pandas_object(rows.candidate_id.astype(str).sort_values(kind='stable'),index=False).values.tobytes()).hexdigest()
 payload={'schema':'native12_failure_attribution_provenance_v2','status':'IMMUTABLE_DIAGNOSTIC_ONLY','rows':len(rows),'candidate_id_sha256':ident,'source_manifest_sha256':sha(SRC/'manifest.json'),'output_sha256':{p.name:sha(p) for p in sorted(SRC.glob('*.parquet'))},'causality':'scores frozen; transition/path joins ex-post diagnostic only; pre-entry contrasts are signal-time fields'}
 OUT.mkdir(parents=True);tmp=OUT/'manifest.json.tmp';tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,OUT/'manifest.json')
if __name__=='__main__':main()
