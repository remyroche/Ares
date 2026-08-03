from __future__ import annotations
import hashlib,json,os,platform,sys,tempfile
from pathlib import Path
import numpy,pandas,sklearn
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data_perp/artifacts';J=A/'pre2026_joint_score_context_incremental_gate_20260730_v2';O=A/'pre2026_joint_score_context_incremental_gate_environment_20260730_v1'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 if O.exists():raise RuntimeError(O)
 if h(J/'manifest.json')!=(J/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed')
 d=O.parent/('.'+O.name+'.tmp');d.mkdir();c={'schema':'joint_v2_environment_provenance_v1','joint_manifest_sha256':h(J/'manifest.json'),'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,'script_hashes':{str((ROOT/'scripts/run_pre2026_joint_score_context_gate.py').resolve()):h(ROOT/'scripts/run_pre2026_joint_score_context_gate.py'),str((ROOT/'scripts/run_pre2026_model_failure_incremental_value.py').resolve()):h(ROOT/'scripts/run_pre2026_model_failure_incremental_value.py')}}; (d/'provenance.json').write_text(json.dumps(c,indent=2,sort_keys=True)+'\n');m={'schema':c['schema'],'status':'SEALED_JOINT_V2_ENVIRONMENT_PROVENANCE_NON_PROMOTION','provenance':c,'outputs_sha256':{'provenance.json':h(d/'provenance.json')}};(d/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');(d/'manifest.sha256').write_text(f'{h(d/"manifest.json")}  manifest.json\n');os.replace(d,O);print(O)
if __name__=='__main__':run()
