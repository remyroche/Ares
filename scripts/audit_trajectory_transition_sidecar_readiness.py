#!/usr/bin/env python3
"""Fail-closed readiness audit for trajectory-only transition stack inputs."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';OUT=ART/'trajectory_transition_sidecar_readiness_20260730_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 candidates=[p for p in ART.glob('**/*.parquet') if 'trajectory' in p.as_posix().lower()]
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  rep={'schema':'trajectory_transition_sidecar_readiness_v1','status':'SEALED_FAIL_CLOSED_TRAJECTORY_SIDECAR_UNAVAILABLE','promotion_eligible':False,'decision_cadence_required':'1h','exact_replay_bar_cadence':'1m_labels_only','found_parquet_candidates':[str(p) for p in candidates],'required_authority':{'historical':'pre-2026 hourly blocked-OOF trajectory probability plus uncertainty, one row per source_utc, labels resolved before each fit cutoff','forward':'frozen untouched-2026 hourly trajectory probability plus uncertainty, one row per source_utc, no 2026 tuning','provenance':'manifest checksum, cadence audit, train_end_exclusive and fit_label_resolution_max fields','prohibited':'type/component/cluster IDs and 1m model rows'},'authorized_action':'wait for the recurrence agent to seal an authoritative sidecar; do not construct substitute trajectory scores or run stack arms'};dump(stage/'report.json',rep);files=[p for p in stage.iterdir() if p.is_file()];m={**rep,'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
