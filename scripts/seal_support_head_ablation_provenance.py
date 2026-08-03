#!/usr/bin/env python3
"""Publish immutable correction/seal sidecars for support-head research artifacts."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SLOPE=ROOT/'data_perp/artifacts/febapr2025_historical_future_slope_fixed_geometry_oof_20260730_v1'
OLD=ROOT/'data_perp/artifacts/bounded_direct_auxiliary_contribution_ablation_20260730_v1'
NEW=ROOT/'data_perp/artifacts/bounded_robust_auxiliary_contribution_ablation_20260730_v2'
def h(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 sm=json.loads((a.slope/'manifest.json').read_text());folds=sorted((a.slope/'folds').rglob('*.joblib'))
 if len(folds)!=2 or not (a.slope/'oof_predictions.parquet').is_file():raise ValueError('missing slope fold/ledger contract')
 if not sm['fingerprint'].get('fixed_geometry') or sm['roles']['future_slope_atr_per_hour.diagnostic']['label_availability']['target_valid_rows']<=0:raise ValueError('slope manifest missing fixed geometry or label availability')
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent))
 slope={'schema':'strict_oof_future_slope_detached_seal_v1','status':'SEALED','artifact':str(a.slope),'artifact_manifest_sha256':h(a.slope/'manifest.json'),'oof_predictions_sha256':h(a.slope/'oof_predictions.parquet'),'folds_sha256':{str(p.relative_to(a.slope)):h(p) for p in folds},'runner_sha256':h(ROOT/'scripts/run_febapr2025_historical_auxiliary_oof.py'),'model_geometry':'fixed per-side production slope geometry; no FS/HPO','feature_contract_sha256':hashlib.sha256('\n'.join(sm['fingerprint']['feature_universe']).encode()).hexdigest(),'label_files':sm['fingerprint']['label_files'],'label_availability':sm['roles']['future_slope_atr_per_hour.diagnostic']['label_availability'],'strict_identity':sm['input_contract']['strict_residual_oof']}
 correction={'schema':'bounded_direct_auxiliary_contribution_ablation_v1_invalidation_v1','status':'INVALID_NONAUTHORITATIVE','invalidated_artifact':str(a.old),'invalidated_manifest_sha256':h(a.old/'manifest.json'),'replacement':str(a.new),'replacement_manifest_sha256':h(a.new/'manifest.json'),'reasons':['control was raw direct_q25 rather than v2 robust_decomposed','March weight selection evaluated an isotonic map on the same March OOF labels used to fit it'],'scope':'v1 must not be interpreted, promoted, or compared as the requested robust-control ablation'}
 write(st/'slope_detached_seal.json',slope);write(st/'v1_invalidation.json',correction);man={'schema':'support_head_provenance_sidecars_v1','status':'SEALED','outputs_sha256':{'slope_detached_seal.json':h(st/'slope_detached_seal.json'),'v1_invalidation.json':h(st/'v1_invalidation.json')},'sources':{'slope_manifest':h(a.slope/'manifest.json'),'v1_manifest':h(a.old/'manifest.json'),'v2_manifest':h(a.new/'manifest.json')}};write(st/'manifest.json',man);(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--slope',type=Path,default=SLOPE);p.add_argument('--old',type=Path,default=OLD);p.add_argument('--new',type=Path,default=NEW);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
