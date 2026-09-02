#!/usr/bin/env python3
"""Supersede the H2 bridge audit after the sealed July common-30 bridge."""
from __future__ import annotations
import hashlib, json, os, shutil, tempfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"data_perp/artifacts"
STACK=ART/"final_identical_row_regime_stack_gam_ablation_20260730_v3"
JULY=ART/"july2025_common30_final_base_residual_oof_bridge_20260730_v1"
MAP=ART/"july_common30_baseline_map_refresh_20260730_v1"
OUT=ART/"h2_2025_identical_row_oof_bridge_audit_20260730_v4"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def sealed(root, manifest='manifest.json'):
 p=Path(root)/manifest;m=Path(root)/'manifest.sha256'
 if not p.is_file() or not m.is_file() or m.read_text().split(maxsplit=1)[0]!=sha(p):raise RuntimeError(f'unsealed {root}')
 return json.loads(p.read_text())
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(f'immutable output exists: {output}')
 stack=sealed(STACK); july=sealed(JULY); refresh=sealed(MAP)
 jc=json.loads((JULY/'bridge_contract.json').read_text());mc=json.loads((MAP/'contract.json').read_text())
 j=pd.read_parquet(JULY/'oof_predictions.parquet',columns=['candidate_id','__ts__','side_name','score_base_alpha','score_residual_expected_ev','execution_label_end_utc','residual_is_oof'])
 j['__ts__']=pd.to_datetime(j.__ts__,utc=True);j['execution_label_end_utc']=pd.to_datetime(j.execution_label_end_utc,utc=True)
 if len(j)!=44640 or j.candidate_id.nunique()!=44640 or not j.residual_is_oof.all() or j[['score_base_alpha','score_residual_expected_ev']].isna().any().any():raise RuntimeError('July bridge score pair check failed')
 old_end=pd.Timestamp(mc['old_label_end_max']);new_end=pd.Timestamp(mc['july_label_end_max'])
 source=pd.DataFrame([
  {'period':'2025-07','availability':'available_sealed_common30','rows':len(j),'base_residual_pair':True,'strict_score_fit_provenance':True,'hourly_candidate_clock':True,'exact_1m_execution_label':True,'scope':'frozen 30-asset common universe; not identical to wider v3 population','use':'baseline-map sensitivity only; non-promotional'},
  {'period':'2025-08_to_2025-11','availability':'unavailable','rows':0,'base_residual_pair':False,'strict_score_fit_provenance':False,'hourly_candidate_clock':False,'exact_1m_execution_label':True,'scope':'common30 execution-label ledgers only','use':'none; do not append or substitute'},
  {'period':'2025-12','availability':'incomplete','rows':0,'base_residual_pair':False,'strict_score_fit_provenance':False,'hourly_candidate_clock':False,'exact_1m_execution_label':False,'scope':'incomplete','use':'none'}])
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  source.to_csv(stage/'source_compatibility_ledger.csv',index=False)
  report={'status':'JULY_COMMON30_AVAILABLE_AUGUST_NOVEMBER_UNAVAILABLE','promotion_eligible':False,'model_sample_cadence':'1h','assessment_sample_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','july':{'rows':int(len(j)),'both_side_rows':j.side_name.value_counts().to_dict(),'start_utc':j.__ts__.min(),'end_utc':j.__ts__.max(),'label_end_max_utc':j.execution_label_end_utc.max(),'base_residual_pair':['score_base_alpha','score_residual_expected_ev'],'strict_residual_oof':True,'bridge_status':jc['status']},'map_age':{'old_last_label_end_utc':old_end,'july_last_label_end_utc':new_end,'reduction_days':float((new_end-old_end)/pd.Timedelta(days=1)),'limitation':'the age reduction is based on a July common30 cohort, not a wider-population identical-row extension'},'remaining_blockers':['August-November have exact 1m-derived economics labels but no compatible hourly base+residual OOF score pair, strict score-fit provenance and PIT feature lineage','July common30 selection differs from the wider frozen v3 candidate population; use only the sealed baseline map sensitivity and do not promote/rewrite context arms','December remains incomplete'],'authorized_next_use':'baseline-only causal map sensitivity on the unchanged 2026 hourly frozen candidate ledger; no context-arm extension without compatible July raw context OOF'}
  dump(stage/'readiness_report.json',report)
  files=[x for x in stage.iterdir() if x.is_file()];manifest={'schema':'h2_2025_identical_row_oof_bridge_audit_v4','status':'SEALED_JULY_COMMON30_READINESS_AUDIT_NON_PROMOTION','promotion_eligible':False,'model_sample_cadence':'1h','assessment_sample_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','no_2026_fit_or_map_labels_used':True,'supersedes':'h2_2025_identical_row_oof_bridge_audit_20260730_v3','inputs':{str((STACK/'manifest.json').resolve()):sha(STACK/'manifest.json'),str((JULY/'manifest.json').resolve()):sha(JULY/'manifest.json'),str((MAP/'manifest.json').resolve()):sha(MAP/'manifest.json')},'outputs_sha256':{x.name:sha(x) for x in files}};dump(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
