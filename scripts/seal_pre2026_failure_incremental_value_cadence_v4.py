#!/usr/bin/env python3
"""Seal hourly cadence proof for immutable pre-2026 failure/value v3."""
from __future__ import annotations
import hashlib, json, os, shutil, tempfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data_perp/artifacts'
SRC=ART/'pre2026_oof_model_failure_incremental_value_20260730_v3'
OUT=ART/'pre2026_oof_model_failure_incremental_value_20260730_v4'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists(): raise RuntimeError(output)
 if sha(SRC/'manifest.json') != (SRC/'manifest.sha256').read_text().split()[0]: raise RuntimeError('unsealed v3')
 x=pd.read_parquet(SRC/'materialized_targets.parquet',columns=['candidate_id','__ts__','execution_label_end_utc','source','era'])
 x.__ts__=pd.to_datetime(x.__ts__,utc=True);x.execution_label_end_utc=pd.to_datetime(x.execution_label_end_utc,utc=True)
 x['literal_id_1h']=x.candidate_id.astype(str).str.contains('|1h|',regex=False);x['timestamp_hour_aligned']=x.__ts__.dt.minute.eq(0)&x.__ts__.dt.second.eq(0);x['pre2026_label']=x.execution_label_end_utc.lt(pd.Timestamp('2026-01-01',tz='UTC'))
 if not x.timestamp_hour_aligned.all() or not x.pre2026_label.all(): raise RuntimeError('fail closed cadence')
 audit=x.groupby(['source','era'],as_index=False).agg(candidate_rows=('candidate_id','size'),literal_candidate_id_1h_fraction=('literal_id_1h','mean'),all_timestamps_hour_aligned=('timestamp_hour_aligned','all'),all_labels_end_before_2026=('pre2026_label','all'),candidate_start_utc=('__ts__','min'),candidate_end_utc=('__ts__','max'),label_end_max_utc=('execution_label_end_utc','max'))
 audit['candidate_cadence_evidence']=audit.source.map(lambda s: 'literal_candidate_id_1h' if s!='blocked_oof_panel' else 'hashed_legacy_identity_plus_sealed_source_1h_contract')
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  audit.to_csv(stage/'cadence_provenance_audit.csv',index=False)
  c={'schema':'pre2026_oof_model_failure_incremental_value_v4_cadence_provenance','status':'SEALED_HOURLY_CADENCE_PROVENANCE_SUPPLEMENT_NON_PROMOTION','promotion_eligible':False,'supersedes_for_cadence_only':'pre2026_oof_model_failure_incremental_value_20260730_v3','decision_cadence':'1h','rule':'All training, leave-era OOF, mapping and candidate decision rows are exact 1h, verified by UTC hour alignment. Newer candidate IDs also contain literal |1h|. The legacy blocked-OOF panel has hashed candidate IDs, so its 1h cadence is established by the sealed source contract plus hourly timestamps. 1m data is nested existing exact-12h label/economics evidence only; it is never a model row.','v3_result_status':'unchanged; no refit, score, target, economics or 2026 application was performed by this supplement'}
  dump(stage/'contract.json',c); files=[p for p in stage.iterdir() if p.is_file()]
  m={'schema':c['schema'],'status':c['status'],'promotion_eligible':False,'contract':c,'counts':{'audited_rows':len(x),'literal_candidate_id_1h_rows':int(x.literal_id_1h.sum()),'all_timestamps_hour_aligned':bool(x.timestamp_hour_aligned.all()),'all_labels_end_before_2026':bool(x.pre2026_label.all())},'inputs_sha256':{str((SRC/'manifest.json').resolve()):sha(SRC/'manifest.json'),str((SRC/'materialized_targets.parquet').resolve()):sha(SRC/'materialized_targets.parquet')},'outputs_sha256':{p.name:sha(p) for p in files}}
  dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception: shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__': print(run())
