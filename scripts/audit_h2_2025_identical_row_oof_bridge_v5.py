#!/usr/bin/env python3
"""H2 readiness v5: July and Aug-Nov bridges sealed; December incomplete."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';J=ART/'july2025_common30_final_base_residual_oof_bridge_20260730_v1';A=ART/'augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1';OUT=ART/'h2_2025_identical_row_oof_bridge_audit_20260730_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 jm=json.loads((J/'bridge_contract.json').read_text());am=json.loads((A/'bridge_contract.json').read_text());jp=J/'oof_predictions.parquet';ap=A/'oos_predictions.parquet'
 if jm['outputs'][jp.name]!=sha(jp) or am['outputs_sha256'][ap.name]!=sha(ap):raise RuntimeError('bridge hashes')
 j=pd.read_parquet(jp,columns=['candidate_id','__ts__','side_name','execution_label_end_utc','residual_is_oof']);a=pd.read_parquet(ap,columns=['candidate_id','__ts__','side_name','execution_label_end_utc','residual_is_oos'])
 for x in [j,a]:x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['execution_label_end_utc']=pd.to_datetime(x.execution_label_end_utc,utc=True)
 if len(j)!=44640 or len(a)!=175680 or j.candidate_id.duplicated().any() or a.candidate_id.duplicated().any() or not j.residual_is_oof.all() or not a.residual_is_oos.all():raise RuntimeError('coverage')
 source=pd.DataFrame([{'period':'2025-07','rows':len(j),'availability':'sealed_common30_blocked_oof','score_lineage':'strict OOF','candidate_scope':'common30'},{'period':'2025-08_to_2025-11','rows':len(a),'availability':'sealed_common30_frozen_july_oos','score_lineage':'frozen pre-Aug OOS','candidate_scope':'common30'},{'period':'2025-12','rows':0,'availability':'incomplete','score_lineage':'none','candidate_scope':'none'}])
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  source.to_csv(stage/'source_compatibility_ledger.csv',index=False);report={'status':'JULY_AND_AUGNOV_COMMON30_AVAILABLE_DECEMBER_INCOMPLETE','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','supersedes':'h2_2025_identical_row_oof_bridge_audit_20260730_v4','available':{'july_rows':len(j),'augnov_rows':len(a),'label_end_max_utc':a.execution_label_end_utc.max(),'scope':'both are common30, not population-identical v3 extensions'},'remaining_blockers':['December incomplete','common30 scope prevents treating H2 as a full v3 identical-row replacement or promotion evidence'],'authorized_use':'non-promotional common30 H2 sensitivity and diagnostics only; maps/models must retain explicit population-mismatch provenance'};dump(stage/'readiness_report.json',report);files=[p for p in stage.iterdir() if p.is_file()];m={'schema':'h2_2025_identical_row_oof_bridge_audit_v5','status':'SEALED_H2_COMMON30_READINESS_AUDIT_NON_PROMOTION','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','inputs':{str((J/'manifest.json').resolve()):sha(J/'manifest.json'),str((A/'manifest.json').resolve()):sha(A/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
