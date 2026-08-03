from pathlib import Path
import hashlib,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.materialize_execution_entry_timing_1m_paths import _manifest_hash
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
labels=ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet';base=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet';parity=ROOT/'data_perp/artifacts/deployed_policy_label_parity_20260727_v1/evidence_gate.json';src=json.load(open(ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/manifest.json'))
a=pd.read_parquet(labels,columns=['candidate_id']);b=pd.read_parquet(base,columns=['candidate_id'])
if len(a)!=509868 or not set(b.candidate_id).issubset(set(a.candidate_id)):raise RuntimeError('accepted label/top40 bind failed')
out=ROOT/'data_perp/artifacts/febapr2025_execution_ev_signed_v2_wrapper_20260727_v1';out.mkdir()
p={'schema':'execution_ev_12h_hourly_policy_labels_v2','prediction_role':'execution_ev_12h_labels','wrapper_only_no_label_mutation':True,'labels':{'path':str(labels),'sha256':sha(labels),'rows':len(a)},'accepted_top40':{'path':str(base),'sha256':sha(base),'rows':len(b),'subset_of_labels':True},'parity_gate':{'path':str(parity),'sha256':sha(parity),'passed':json.load(open(parity))['comparison']['parity_pass']},'source_manifest':{'path':str(ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/manifest.json'),'sha256':sha(ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/manifest.json')},'timing':src['timing'],'exit_policy_contract':src['exit_policy_contract'],'accounting':src['accounting'],'geometry':src['geometry'],'targets':src['targets']}
p['prediction_role_manifest_sha256']=_manifest_hash(p);(out/'manifest.json').write_text(json.dumps(p,indent=2,sort_keys=True))
