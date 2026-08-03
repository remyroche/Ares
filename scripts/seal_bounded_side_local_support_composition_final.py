#!/usr/bin/env python3
import argparse,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'data_perp/artifacts/bounded_side_local_support_composition_20260730_v1';CORR=ROOT/'data_perp/artifacts/bounded_side_local_support_composition_20260730_v2_tie_correction'
def h(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 files={'v1_manifest':a.v1/'manifest.json','correction_manifest':a.correction/'manifest.json','composition_runner':ROOT/'scripts/run_bounded_side_local_support_composition.py','correction_runner':ROOT/'scripts/correct_bounded_side_local_support_composition_ties.py','composition_test':ROOT/'tests/test_bounded_side_local_support_composition.py','correction_test':ROOT/'tests/test_correct_bounded_side_local_support_composition_ties.py'}
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));payload={'schema':'bounded_side_local_support_composition_final_seal_v1','status':'SEALED_CORRECTED_NONPROMOTION','files_sha256':{k:h(v) for k,v in files.items()},'corrected_tie_bounds_sha256':h(a.correction/'corrected_tie_bounds.csv'),'strict_adverse_proof_sha256':h(a.correction/'adverse_strict_oof_proof.json'),'test_contract':'focused tests must pass: global paired grid/adverse support, control parity/cutoff/ties, mixed-sign expected precision, strict adverse fold proof'};(st/'seal.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');m={'schema':'bounded_side_local_support_composition_final_seal_v1','status':'SEALED_CORRECTED_NONPROMOTION','seal_sha256':h(st/'seal.json'),'inputs':payload['files_sha256'],'runner_and_tests_bound':True};(st/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--v1',type=Path,default=V1);p.add_argument('--correction',type=Path,default=CORR);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
