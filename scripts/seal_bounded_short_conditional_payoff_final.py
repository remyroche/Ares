#!/usr/bin/env python3
import argparse,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2';DAY=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2_day_blocks'
def h(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 files={'ablation_manifest':a.ablation/'manifest.json','day_block_manifest':a.day_blocks/'manifest.json','ablation_runner':ROOT/'scripts/run_bounded_short_conditional_payoff_ablation.py','day_block_runner':ROOT/'scripts/report_bounded_short_conditional_payoff_day_blocks.py','focused_test':ROOT/'tests/test_bounded_short_conditional_payoff_ablation.py','tie_formula_test':ROOT/'tests/test_correct_bounded_side_local_support_composition_ties.py'}
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));seal={'schema':'bounded_short_conditional_payoff_final_seal_v1','status':'SEALED_NONPROMOTION','files_sha256':{k:h(v) for k,v in files.items()},'test_contract':'focused test asserts runner hash, frozen A-control parity, March-only selection, April population, corrected mixed-tie precision, gate failure, and fixed-book day-block coverage/CI'};(st/'seal.json').write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n');m={'schema':'bounded_short_conditional_payoff_final_seal_v1','status':'SEALED_NONPROMOTION','seal_sha256':h(st/'seal.json'),'runner_and_tests_bound':True,'inputs':seal['files_sha256']};(st/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ablation',type=Path,default=OUT);p.add_argument('--day-blocks',type=Path,default=DAY);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
