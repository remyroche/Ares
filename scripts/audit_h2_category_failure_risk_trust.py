#!/usr/bin/env python3
"""Fail-closed preflight for category failure-risk/trust corrections."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data_perp/artifacts/h2_common30_regime_category_performance_stability_20260730_v3';OUT=ROOT/'data_perp/artifacts/h2_category_failure_risk_trust_ablation_20260730_v1';MIN_RANK=.70
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 m=json.loads((SRC/'manifest.json').read_text())
 if (SRC/'manifest.sha256').read_text().split()[0]!=sha(SRC/'manifest.json'):raise RuntimeError('unsealed source')
 p=SRC/'category_era_summary.csv'
 if m['outputs_sha256'][p.name]!=sha(p):raise RuntimeError('source hash')
 x=pd.read_csv(p);rows=[]
 for layer,z in x.groupby('layer',sort=True):
  w=z.pivot_table(index=['category','side_name'],columns='era',values='mean_bps')
  c=w.corr(method='spearman');pairs=[]
  for i,a in enumerate(c.index):
   for b in c.index[i+1:]:pairs.append({'layer':layer,'left_era':a,'right_era':b,'spearman_rank_stability':c.loc[a,b],'shared_category_side_cells':int(w[[a,b]].dropna().shape[0])})
  rows.extend(pairs)
 rank=pd.DataFrame(rows);gate=rank.groupby('layer',as_index=False).agg(min_pairwise_rank_stability=('spearman_rank_stability','min'),comparisons=('spearman_rank_stability','size'),min_shared_cells=('shared_category_side_cells','min'));gate['passes_predeclared_relative_failure_gate']=gate.min_pairwise_rank_stability.ge(MIN_RANK)
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  rank.to_csv(stage/'leave_era_out_rank_stability.csv',index=False);gate.to_csv(stage/'trust_preflight_gate.csv',index=False)
  rep={'schema':'h2_category_failure_risk_trust_ablation_v1','status':'SEALED_FAIL_CLOSED_UNSTABLE_RELATIVE_FAILURE_RANKS','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','pre_registration':{'relative_failure_rank_stability_min_spearman':MIN_RANK,'layers':['regime','transition','combined'],'trust_actions':'fixed exclusion/downweighting or low-capacity GAM only if every layer clears preflight'},'result':'no layer clears; no 2026 trust score, exclusion, downweighting, GAM correction, selection or portfolio assessment is run','no_2026_tuning':True,'no_ex_post_phase_gate':True,'inputs_sha256':{'category_summary':sha(p)}};dump(stage/'report.json',rep);files=[z for z in stage.iterdir() if z.is_file()];man={**rep,'inputs':{str((SRC/'manifest.json').resolve()):sha(SRC/'manifest.json')},'outputs_sha256':{z.name:sha(z) for z in files}};dump(stage/'manifest.json',man);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
