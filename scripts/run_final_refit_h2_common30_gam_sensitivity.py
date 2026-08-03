#!/usr/bin/env python3
"""Final fixed-GAM refit with common-30 H2 labels; 2026 sensitivity only."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_final_identical_row_regime_stack_gam_ablation as final
from scripts import run_july_common30_baseline_map_refresh as base
from scripts.run_augnov2025_common30_context_oos_extension import join_context,sha
IDENTITY=('candidate_id','__ts__','__symbol__','side_name');TARGET=final.TARGET;CUT=pd.Timestamp('2026-01-01',tz='UTC')
SIDE=ROOT/'data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1/manifest.json';PANEL=ROOT/'data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet';STACK=ROOT/'data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3';JULY=ROOT/'data_perp/artifacts/july2025_common30_regime_context_raw_score_extension_20260730_v1';AUG=ROOT/'data_perp/artifacts/augnov2025_common30_fixed_preaug_context_oos_extension_20260730_v2';AUGBR=ROOT/'data_perp/artifacts/augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1';OUT=ROOT/'data_perp/artifacts/final_refit_h2_common30_gam_sensitivity_20260730_v2'
class Err(RuntimeError):pass
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def seal(root,schema,status):
 root=Path(root);m=root/'manifest.json';k=root/'manifest.sha256'
 if not k.is_file() or k.read_text().split()[0]!=sha(m):raise Err(f'unsealed {root}')
 d=json.loads(m.read_text())
 if d.get('schema')!=schema or d.get('status')!=status:raise Err(f'wrong {root}')
 return d
def raw_file(root,name,expected):
 d=seal(root,*expected);p=Path(root)/name
 if d.get('outputs_sha256',{}).get(name)!=sha(p):raise Err(f'hash {p}')
 return pd.read_parquet(p),d
def forward_join(x,ctx):
 # The frozen forward ledger retains a copy of old context values for its
 # historical arm record.  The sealed sidecar is the authority for this
 # refit; remove overlaps before the many-to-one join rather than accepting
 # pandas' suffixed duplicate fields.
 x=x.drop(columns=[c for c in ctx.columns if c!='source_utc' and c in x.columns],errors='ignore').copy()
 before=x[list(IDENTITY)].sort_values(list(IDENTITY),kind='stable').reset_index(drop=True);o=x.merge(ctx,left_on='__ts__',right_on='source_utc',how='left',validate='many_to_one');after=o[list(IDENTITY)].sort_values(list(IDENTITY),kind='stable').reset_index(drop=True)
 if not before.equals(after) or o.source_utc.isna().any():raise Err('forward context identity')
 if not o.bocpd_regime_available.astype(bool).all() or not o.lgbm_transition_available.astype(bool).all():raise Err('forward context availability')
 for s in ['bocpd','lgbm']:
  if not o[f'provenance_partition_{s}'].eq('untouched_2026_forward').all():raise Err('forward provenance')
 return o.drop(columns='source_utc')
def map_fit(stack_hist,july_raw,aug_raw,stack_arm,july_arm,aug_arm):
 a=stack_hist[stack_hist.arm.eq(stack_arm)][list(IDENTITY)+['raw_score',TARGET,'execution_label_end_utc']]
 b=july_raw[july_raw.arm.eq(july_arm)][list(IDENTITY)+['raw_score',TARGET,'execution_label_end_utc']]
 c=aug_raw[aug_raw.arm.eq(aug_arm)][list(IDENTITY)+['raw_score',TARGET,'execution_label_end_utc']]
 o=pd.concat([a,b,c],ignore_index=True);o['__ts__']=pd.to_datetime(o.__ts__,utc=True);o['execution_label_end_utc']=pd.to_datetime(o.execution_label_end_utc,utc=True)
 if o.duplicated(IDENTITY).any() or not o.execution_label_end_utc.lt(CUT).all():raise Err(f'map source {stack_arm}')
 return o
def run(output=OUT):
 output=Path(output)
 if output.exists():raise Err(output)
 _,rp,tp=final._load_manifest(SIDE);ctx=final._hourly_context(rp,tp)
 hist=final._join(final._verified_scores(PANEL,role='historical'),ctx,role='historical')
 # All newly available pre-2026 labelled residual rows join the same causal context authority.
 jbridge=json.loads((ROOT/'data_perp/artifacts/july2025_common30_final_base_residual_oos_bridge_20260730_v1/bridge_contract.json').read_text()) if False else None
 jpred=ROOT/'data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1/oof_predictions.parquet';jc=json.loads((jpred.parent/'bridge_contract.json').read_text())
 if jc['outputs'][jpred.name]!=sha(jpred):raise Err('July bridge hash')
 j=pd.read_parquet(jpred,columns=[*IDENTITY,TARGET,final.RESIDUAL,'execution_label_end_utc']);j['__ts__']=pd.to_datetime(j.__ts__,utc=True);j['execution_label_end_utc']=pd.to_datetime(j.execution_label_end_utc,utc=True);j=join_context(j,ctx,'July refit train')
 apred=AUGBR/'oos_predictions.parquet';am=seal(AUGBR,'augnov2025_common30_frozen_july_base_residual_oos_bridge_v1','SEALED_COMMON30_FROZEN_JULY_OOS_SCORE_BRIDGE_NON_PROMOTION')
 if am['outputs_sha256'][apred.name]!=sha(apred):raise Err('H2 bridge hash')
 a=pd.read_parquet(apred);a['__ts__']=pd.to_datetime(a.__ts__,utc=True);a['execution_label_end_utc']=pd.to_datetime(a.execution_label_end_utc,utc=True);a=join_context(a,ctx,'H2 refit train')
 train=pd.concat([hist,j,a],ignore_index=True,sort=False);train=train[train.execution_label_end_utc.lt(CUT)].copy()
 if not train.execution_label_end_utc.lt(CUT).all():raise Err('2026 label in refit')
 # Existing raw OOF/OOS score ledgers are the only calibration inputs.
 sh,sm=raw_file(STACK,'historical_oof_scores.parquet',('final_identical_row_regime_stack_gam_ablation_v3','SEALED_STRICT_FORWARD_IDENTICAL_ROW_ABLATION_NON_PROMOTION'))
 jr,jm=raw_file(JULY,'july_raw_context_scores.parquet',('july2025_common30_regime_context_raw_score_extension_v1','SEALED_STRICT_PREJULY_TRAINED_JULY_COMMON30_RAW_CONTEXT_EXTENSION_NON_PROMOTION'))
 ar,am2=raw_file(AUG,'augnov_raw_context_scores.parquet',('augnov2025_common30_fixed_preaug_context_oos_extension_v2','SEALED_FIXED_PREAUG_CONTEXT_OOS_EXTENSION_NON_PROMOTION'))
 # The context extension deliberately has no baseline arm.  Bind its frozen
 # residual OOS score from the validated bridge as the baseline map cohort.
 ar=pd.concat([ar,a[[*IDENTITY,final.RESIDUAL,TARGET,'execution_label_end_utc']].rename(columns={final.RESIDUAL:'raw_score'}).assign(arm='score_residual_expected_ev')],ignore_index=True,sort=False)
 fm=seal(STACK,'final_identical_row_regime_stack_gam_ablation_v3','SEALED_STRICT_FORWARD_IDENTICAL_ROW_ABLATION_NON_PROMOTION');fp=STACK/'frozen_2026_candidate_scores.parquet'
 if fm['outputs_sha256'][fp.name]!=sha(fp):raise Err('forward hash')
 f=pd.read_parquet(fp,filters=[('arm','==','baseline')]);f['__ts__']=pd.to_datetime(f.__ts__,utc=True);f['execution_label_end_utc']=pd.to_datetime(f.execution_label_end_utc,utc=True);f=forward_join(f,ctx)
 arms=[('baseline_frozen','baseline','baseline_raw_residual','score_residual_expected_ev',None)]
 for c in ['regime','transition','combined']:arms.append((f'final_refit_gam_{c}',f'gam_{c}_only',f'additive_bounded_gam_{c}_raw',f'additive_bounded_gam_{c}',final.Arm(f'gam_{c}','additive_bounded_gam',c,TARGET,'gam')))
 rows=[];periods=[];sides=[];cal=[];selected=[];audit=[]
 for n,(name,stack_arm,july_arm,aug_arm,model_arm) in enumerate(arms):
  mf=map_fit(sh,jr,ar,stack_arm,july_arm,aug_arm)
  mapper=base._fit(mf);pieces=[]
  for side,x in f.groupby('side_name',observed=True,sort=True):
   fit=train[train.side_name.eq(side)]
   if model_arm is None:raw=x[final.RESIDUAL].to_numpy(float);meta={'family':'frozen_residual'}
   else:raw,meta=final._predict(fit,x,model_arm,202600+n*17+(side=='short'))
   pieces.append(x.assign(arm=name,raw_score=raw));audit.append({'arm':name,'side_name':side,'model_fit_rows':len(fit),'model_fit_label_end_max':fit.execution_label_end_utc.max(),'map_fit_rows':len(mf),'map_fit_label_end_max':mf.execution_label_end_utc.max(),'no_2026_label_fit':True,**meta})
  score=pd.concat(pieces,ignore_index=True)
  for strict in [False,True]:
   map_name=f'{name}__{"rank_preserving" if strict else "isotonic"}';z=score.copy();m=mapper.predict(z.raw_score.to_numpy(float));z['mapped_score']=base._strict_rank(m,z.raw_score.to_numpy(float)) if strict else m;summary,per,side,cc=base._evaluate(z,map_name);summary.update({'arm':name,'map_method':map_name,'rank_preserving':strict,'h2_common30_map_rows':len(mf)});rows.append(summary);periods.append(per);sides.append(side);cal.append(cc);z=base._select(z);selected.append(z[z.selected_global_top10][list(IDENTITY)+['arm','raw_score','mapped_score',TARGET,base.GROSS,base.COST,'execution_label_end_utc','selected_global_top10']].assign(map_method=map_name))
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  pd.DataFrame(rows).to_csv(stage/'metrics_summary.csv',index=False);pd.concat(periods).to_parquet(stage/'period_metrics.parquet',index=False);pd.concat(sides).to_parquet(stage/'side_metrics.parquet',index=False);pd.concat(cal).to_parquet(stage/'calibration_deciles.parquet',index=False);pd.concat(selected).to_parquet(stage/'frozen_2026_selected_scores.parquet',index=False);dump(stage/'fit_audit.json',audit)
  contract={'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','models':'fixed bounded-GAM regime/transition/combined refit side-locally on all compatible pre-2026 labels; baseline frozen residual control','map':'increasing and rank-preserving isotonic only on concatenated v3 OOF + July OOF + Aug-Nov OOS raw scores; no 2026 label','assessment':'unchanged 127777 2026 hourly candidates, one pooled global top10 per arm/map','no_hpo_or_2026_tuning':True,'scope_limitation':'July/Aug-Nov additions are common30, not population-identical to v3; sensitivity only, no promotion','supersedes_invalid':'final_refit_h2_common30_gam_sensitivity_20260730_v1: only h2_common30_map_rows reporting field was aggregate rather than arm-specific; scores/results unchanged'};dump(stage/'contract.json',contract);files=[p for p in stage.iterdir() if p.is_file()];manifest={'schema':'final_refit_h2_common30_gam_sensitivity_v2','status':'SEALED_H2_COMMON30_FINAL_REFIT_GAM_SENSITIVITY_NON_PROMOTION','promotion_eligible':False,'contract':contract,'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',manifest);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
