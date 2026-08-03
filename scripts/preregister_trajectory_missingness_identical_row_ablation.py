#!/usr/bin/env python3
"""Seal missingness-aware trajectory identical-row ablation before economics."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';S=ART/'hourly_trajectory_transition_soft_sidecar_20260730_v1';V=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3';OUT=ART/'trajectory_missingness_identical_row_ablation_preregistration_20260730_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 for r in [S,V]:
  if (r/'manifest.sha256').read_text().split()[0]!=sha(r/'manifest.json'):raise RuntimeError(f'unsealed {r}')
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  contract={'schema':'trajectory_missingness_identical_row_ablation_preregistration_v1','status':'SEALED_PREREGISTERED_NO_FORWARD_ECONOMICS_READ','decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','row_policy':'retain every historical and frozen-2026 candidate row; no availability-based dropping','trajectory_missingness':{'feature':'trajectory_available','when_unavailable':{'trajectory_transition_probability':0.5,'probability_entropy':0.6931471805599453,'top2_margin':0.0},'imputation':'fixed neutral constants only; no forward-derived imputation'},'fallback':'existing transition context remains present for every row and is never replaced by trajectory fields','arms':['baseline_existing_transition_control','trajectory_availability_neutral_only','existing_transition_plus_trajectory','regime_plus_trajectory','regime_plus_existing_transition_plus_trajectory'],'learner':'side-local low-capacity GAM; fixed feature placements only','fit_map':'pre-2026 OOF only; increasing/rank-preserving maps only','assessment':'unchanged frozen 127777 2026 hourly candidates; one pooled global top10','reports':['availability-stratified metrics','aggregate/latest/week/month Q10/Q50','both-side economics','turnover and concentration'],'prohibited':['cluster/type/component IDs','2026 tuning','1m model rows','ex-post phases'],'promotion_eligible':False};dump(stage/'contract.json',contract);files=[p for p in stage.iterdir() if p.is_file()];m={**contract,'inputs':{str((S/'manifest.json').resolve()):sha(S/'manifest.json'),str((V/'manifest.json').resolve()):sha(V/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
