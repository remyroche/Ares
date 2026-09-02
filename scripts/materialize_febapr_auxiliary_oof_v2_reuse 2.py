#!/usr/bin/env python3
"""Rescore exact residual-OOF identities from asserted equivalent v1 folds."""
import hashlib, json, os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_febapr2025_historical_auxiliary_oof import (DEFAULT_CONTEXT, DEFAULT_LABEL_DIR, DEFAULT_STRICT_RESIDUAL, IDENTITY, OUTER_MONTHS, _identity_sha, _role_metrics, build_role_targets, load_inputs)

V1=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v1'
OUT=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2'
ROLES=('peak_mfe_12h_atr.p_hit','peak_mfe_12h_atr.conditional_mean')
SIDES=('long','short')
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,default=str,indent=2,sort_keys=True));os.replace(t,p)
def main():
 f,features,universe,strict=load_inputs(DEFAULT_CONTEXT,DEFAULT_LABEL_DIR,DEFAULT_STRICT_RESIDUAL); targets=build_role_targets(f,role_names=ROLES)
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(); out=f.loc[f.__strict_residual_oof__].loc[:,list(IDENTITY)+['__decision_ts__','__label_end_ts__','__meaningful_mfe_reached_12h__','__peak_mfe_atr_12h__']].copy()
 report={}
 for role in ROLES:
  pred=np.full(len(f),np.nan,np.float32); rr={}; task=targets[role].role.task
  for month in OUTER_MONTHS:
   cut=pd.Timestamp(pd.Period(month,freq='M').start_time,tz='UTC'); valid=f.__strict_residual_oof__.to_numpy() & f.__ts__.dt.strftime('%Y-%m').eq(month).to_numpy()
   fold={}
   for side in SIDES:
    a=joblib.load(V1/'folds'/role.replace('.','__')/(month+'.joblib'))['models'][side]; ref=(f.side_name.eq(side)&f.__decision_ts__.lt(cut)&f.__label_end_ts__.lt(cut)&targets[role].train_mask&np.isfinite(targets[role].target))
    c=a['reference_split_contract']; assert c['selection_hpo_reference_end']==cut.isoformat() and c['role_reference_rows']==int(ref.sum())
    m=a['oof_models'][0]; rows=np.flatnonzero(valid & f.side_name.eq(side).to_numpy()); X=f.iloc[rows].loc[:,a['selected_features']]
    p=m.predict_proba(X)[:,1] if task=='binary' else m.predict(X); pred[rows]=p
    fold[side]={'selected_features':a['selected_features'],'best_params':a['best_params'],'hpo':a['hpo'],'reference_rows':int(ref.sum()),'model_sha256':a['fold_provenance'][0]['model_sha256']}
   rr[month]=fold
  mask=f.__strict_residual_oof__.to_numpy() & targets[role].train_mask & np.isfinite(targets[role].target)
  report[role]={'task':task,'oof_metrics':_role_metrics(targets[role].target,pred,mask,task_kind=task,quantile_alpha=.8),'folds':rr};out['pred_'+role.replace('.','__')]=pred[f.__strict_residual_oof__.to_numpy()]
 if len(out)!=140682 or _identity_sha(out)!=strict['identity_sha256']: raise AssertionError('strict identity mismatch')
 out.to_parquet(OUT/'oof_predictions.parquet',index=False,compression='zstd'); write(OUT/'manifest.json',{'schema':'febapr_auxiliary_oof_v2_exact_signal_identity_reuse','status':'COMPLETE','strict_residual':strict,'rows':len(out),'roles':report,'v1_reuse':'models only after asserted same reference cutoff/rows; v1 predictions forbidden','remaining_11_roles':'pending_same_strict_side_local_fold_local_fs_hpo'})
if __name__=='__main__': main()
