#!/usr/bin/env python3
"""Seal the invalidation of the pre-fix frozen-band v1 diagnostic."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'data_perp/artifacts/frozen_month_score_band_transition_diagnostic_20260730_v1'
FIXED=ROOT/'data_perp/artifacts/frozen_month_score_band_transition_diagnostic_20260730_v2'
OUT=ROOT/'data_perp/artifacts/frozen_month_score_band_transition_diagnostic_20260730_v1_invalidation'
def h(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def run()->dict:
 if OUT.exists():raise FileExistsError(OUT)
 for p in (OLD/'manifest.json',OLD/'manifest.sha256',FIXED/'manifest.json',FIXED/'manifest.sha256'):
  if not p.is_file():raise FileNotFoundError(p)
 old=json.loads((OLD/'manifest.json').read_text());fixed=json.loads((FIXED/'manifest.json').read_text())
 if (OLD/'manifest.sha256').read_text().split()[0]!=h(OLD/'manifest.json') or (FIXED/'manifest.sha256').read_text().split()[0]!=h(FIXED/'manifest.json'):raise ValueError('source seal fails')
 payload={'schema':'frozen_month_score_band_transition_diagnostic_v1_invalidation','status':'INVALIDATED_PRE_FIX_OUTPUT_DO_NOT_USE','invalidated_artifact':{'path':str(OLD),'manifest_sha256':h(OLD/'manifest.json'),'runner_sha256':old.get('runner',{}).get('sha256')},'reasons':['fixed/global top contribution was emitted only for target_local because the source-frozen loop block was outside the month/scheme loop','v1 lacks required source-frozen source and target contribution coverage'],'replacement_artifact':{'path':str(FIXED),'manifest_sha256':h(FIXED/'manifest.json'),'runner_sha256':fixed.get('runner',{}).get('sha256')},'policy_state':'unchanged; both artifacts are diagnostic only','runner':{'path':str(Path(__file__).resolve()),'sha256':h(Path(__file__).resolve())}}
 OUT.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(dir=OUT.parent,prefix=f'.{OUT.name}.'))
 try:
  (stage/'invalidation.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');manifest={'schema':payload['schema'],'status':payload['status'],'invalidation_sha256':h(stage/'invalidation.json'),'runner':payload['runner']};(stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(h(stage/'manifest.json')+'\n');os.replace(stage,OUT)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
 return payload
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
