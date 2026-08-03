#!/usr/bin/env python3
"""Seal the frozen-2026 correction protocol without reading candidate economics."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts'
V3=ART/'pre2026_oof_model_failure_incremental_value_20260730_v3';V4=ART/'pre2026_oof_model_failure_incremental_value_20260730_v4';CONTROL=ART/'pre2026_joint_score_context_incremental_gate_20260730_v2';ENV=ART/'pre2026_joint_score_context_incremental_gate_environment_20260730_v1';STACK=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3';OUT=ART/'frozen_2026_failure_value_correction_preregistration_20260730_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 for p in [V3,V4,CONTROL,ENV,STACK]:
  if sha(p/'manifest.json') != (p/'manifest.sha256').read_text().split()[0]:raise RuntimeError(f'unsealed prerequisite {p.name}')
 # The stack manifest is provenance only: do not open the 2026 candidate parquet or its outcomes here.
 sm=json.loads((STACK/'manifest.json').read_text())
 gate=json.loads((CONTROL/'manifest.json').read_text());g=__import__('pandas').read_csv(CONTROL/'eligibility.csv');eligible=g[g.eligible].to_dict('records')
 c={'schema':'frozen_2026_failure_value_correction_preregistration_v3','status':'SEALED_PREREGISTERED_NO_2026_ECONOMICS_READ','promotion_eligible':False,'supersedes':['frozen_2026_failure_value_correction_preregistration_20260730_v1','frozen_2026_failure_value_correction_preregistration_20260730_v2'],'implementation_sha256':{str(Path(__file__).resolve()):sha(Path(__file__))},'joint_implementation_sha256':gate['contract']['implementation_sha256'],'prerequisite_artifacts':{'failure_value_v3_manifest_sha256':sha(V3/'manifest.json'),'cadence_v4_manifest_sha256':sha(V4/'manifest.json'),'joint_matched_control_manifest_sha256':sha(CONTROL/'manifest.json'),'joint_environment_manifest_sha256':sha(ENV/'manifest.json'),'frozen_stack_manifest_sha256':sha(STACK/'manifest.json')},'context_incremental_gate':{'source':str((CONTROL/'eligibility.csv').resolve()),'eligible_rows':eligible,'all_context_heads_rejected':len(eligible)==0},'application':{'authorized':False,'authorization_rule':'Joint-v2 has zero eligible arm/target heads; do not open frozen-2026 candidate/economics files or score any correction.'},'prohibited':['2026 read/tuning','severity head','1m model rows','portfolio replay','policy promotion']}
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  dump(stage/'contract.json',c);files=[p for p in stage.iterdir() if p.is_file()];m={'schema':c['schema'],'status':c['status'],'promotion_eligible':False,'contract':c,'frozen_stack_manifest_schema':sm.get('schema'),'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
