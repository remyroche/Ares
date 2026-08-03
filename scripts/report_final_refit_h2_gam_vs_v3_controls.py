#!/usr/bin/env python3
"""Fixed 2026 comparison: sealed H2 GAM refit against frozen v3 GAM controls."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];STACK=ROOT/'data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3';REFIT=ROOT/'data_perp/artifacts/final_refit_h2_common30_gam_sensitivity_20260730_v2';OUT=ROOT/'data_perp/artifacts/final_refit_h2_common30_gam_vs_v3_controls_20260730_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 sm=json.loads((STACK/'manifest.json').read_text());rm=json.loads((REFIT/'manifest.json').read_text())
 if (STACK/'manifest.sha256').read_text().split()[0]!=sha(STACK/'manifest.json') or (REFIT/'manifest.sha256').read_text().split()[0]!=sha(REFIT/'manifest.json'):raise RuntimeError('unsealed source')
 f=STACK/'frozen_2026_candidate_scores.parquet';r=REFIT/'metrics_summary.csv'
 if sm['outputs_sha256'][f.name]!=sha(f) or rm['outputs_sha256'][r.name]!=sha(r):raise RuntimeError('source hash')
 x=pd.read_parquet(f);x['__ts__']=pd.to_datetime(x.__ts__,utc=True);rows=[]
 for arm in ['baseline','gam_regime_only','gam_transition_only','gam_combined']:
  z=x[x.arm.eq(arm)].copy();z=z.sort_values(['mapped_score','raw_score','candidate_id'],ascending=[False,False,True],kind='stable');p=z.head((len(z)+9)//10);rows.append({'source':'frozen_v3_control','arm':arm,'top10_net_ev':p.execution_net_ev_12h.mean(),'execution_rank_ic':z.mapped_score.corr(z.execution_net_ev_12h,method='spearman'),'week_net_ev_q10':None,'month_net_ev_q10':None})
 y=pd.read_csv(r);y=y[y.rank_preserving].copy();y['source']='h2_common30_final_refit';rows.extend(y[['source','arm','top10_net_ev','execution_rank_ic','week_net_ev_q10','month_net_ev_q10']].to_dict('records'))
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  pd.DataFrame(rows).to_csv(stage/'comparison.csv',index=False);contract={'assessment':'same frozen 127777 2026 hourly candidates; comparison only','controls':'original frozen-v3 mapped baseline/GAM controls','refits':'rank-preserving map rows from sealed H2 common30 final-refit sensitivity','no_2026_fit_tuning_or_selection':True,'limitation':'H2 additions are common30 only'};dump(stage/'contract.json',contract);files=[p for p in stage.iterdir() if p.is_file()];m={'schema':'final_refit_h2_gam_vs_v3_controls_v1','status':'SEALED_2026_CONTROL_COMPARISON_NON_PROMOTION','promotion_eligible':False,'contract':contract,'inputs':{str((STACK/'manifest.json').resolve()):sha(STACK/'manifest.json'),str((REFIT/'manifest.json').resolve()):sha(REFIT/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
