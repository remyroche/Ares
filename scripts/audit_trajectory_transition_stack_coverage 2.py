#!/usr/bin/env python3
"""Fail-closed trajectory sidecar coverage audit for identical-row stack."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';S=ART/'hourly_trajectory_transition_soft_sidecar_20260730_v1';V=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3';OUT=ART/'trajectory_transition_identical_row_stack_coverage_20260730_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 sm=json.loads((S/'manifest.json').read_text());vm=json.loads((V/'manifest.json').read_text());sp=S/'hourly_trajectory_transition_soft_sidecar.parquet';hp=V/'historical_oof_scores.parquet';fp=V/'frozen_2026_candidate_scores.parquet'
 if sm['outputs_sha256'][sp.name]!=sha(sp) or vm['outputs_sha256'][hp.name]!=sha(hp) or vm['outputs_sha256'][fp.name]!=sha(fp):raise RuntimeError('hash')
 s=pd.read_parquet(sp);s.source_utc=pd.to_datetime(s.source_utc,utc=True);rows=[]
 for name,p,partition in [('historical',hp,'blocked_era_oof'),('forward_2026',fp,'untouched_2026_frozen_fit')]:
  x=pd.read_parquet(p,columns=['__ts__']);ts=pd.DataFrame({'__ts__':pd.to_datetime(x.__ts__,utc=True).drop_duplicates()});z=ts.merge(s[['source_utc','trajectory_available','provenance_partition']],left_on='__ts__',right_on='source_utc',how='left');rows.append({'partition':name,'candidate_timestamps':len(z),'available_timestamps':int(z.trajectory_available.fillna(False).sum()),'availability_fraction':float(z.trajectory_available.fillna(False).mean()),'expected_provenance':partition,'provenance_ok':bool(z.loc[z.trajectory_available.fillna(False),'provenance_partition'].eq(partition).all()),'unavailable_timestamps':int((~z.trajectory_available.fillna(False)).sum())})
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  pd.DataFrame(rows).to_csv(stage/'timestamp_coverage.csv',index=False);rep={'schema':'trajectory_transition_identical_row_stack_coverage_v1','status':'SEALED_FAIL_CLOSED_FORWARD_TRAJECTORY_COVERAGE_INSUFFICIENT','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','pre_registration':'no missingness-indicator or availability-conditioned stack arm was pre-registered; exact identical rows prohibit drop/fill','action':'do not run trajectory stack arms until a pre-registered missingness policy or complete untouched-2026 trajectory coverage exists'};dump(stage/'report.json',rep);files=[p for p in stage.iterdir() if p.is_file()];m={**rep,'inputs':{str((S/'manifest.json').resolve()):sha(S/'manifest.json'),str((V/'manifest.json').resolve()):sha(V/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
