#!/usr/bin/env python3
"""Bounded matched GMM/DAE/failure-first economic OOF ablation.

The largest currently exact common surface is the May--July 2026 intersection
of the representation handoff and strict-model-OOS failure-first overlay.  It
is deliberately not described as all-era coverage.  Each arm receives its own
chronological side-local predictor and trailing-21-day causal EV map; selection
is one pooled global top-10 only after that map.
"""

from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, validate_candidate_identity  # noqa: E402
from extreme_price_movements.regime_stack_evaluation import EvaluationColumns, evaluate_matched_arms, global_top_k_mask  # noqa: E402

SCHEMA="unsupervised_economic_common_oof_v1"
DEFAULT_REP=ROOT/"data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet"
DEFAULT_FAILURE=ROOT/"data_perp/artifacts/failure_first_detector_current_transfer_20260726_v6/candidate_overlay.parquet"
DEFAULT_LABEL_DIR=ROOT/"data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
DEFAULT_OUTPUT=ROOT/"data_perp/artifacts/unsupervised_economic_common_oof_20260730_v3"
TARGET="execution_net_ev_12h"; ALPHA="__first_touch_target_soft__"; LABEL_DELAY=pd.Timedelta(hours=12)
ACTION_TOKENS=("timing","mae","target_price","wait","action","entry_price")
RETAINED_REPRESENTATION_DIAGNOSTICS=[*[f"dae_b16_{i:02d}" for i in range(16)],"dae_reconstruction_error_zscore","gmm_ood_score","mahalanobis_distance","expected_mahalanobis"]

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def feature_lists()->dict[str,list[str]]:
 base=["catboost__residual__without_hpo__all_features"]
 # Geometry only: posterior, entropy, and compact risk summaries are deliberately
 # excluded until a later, separately-gated increment test establishes value.
 gmm=["gmm_ood_score","mahalanobis_distance","expected_mahalanobis"]
 dae=[*[f"dae_b16_{i:02d}" for i in range(16)],"dae_reconstruction_error_zscore"]
 failure=["p_failure_destination_3h","p_transition_within_3h"] # distinct competing-risk and transition probabilities
 result={"baseline":base,"gmm_only":[*base,*gmm],"dae_only":[*base,*dae],"gmm_plus_dae":[*base,*gmm,*dae],"failure_destination_only":[*base,failure[0]],"failure_transition_only":[*base,failure[1]],"failure_first_context":[*base,*failure]}
 for cols in result.values():
  if any(any(t in c.lower() for t in ACTION_TOKENS) for c in cols): raise RegimeOOFStackError("action field entered unsupervised ablation")
 return result

def _label_paths(label_dir:Path)->list[Path]:
 return [label_dir/f"train_global_{side}_5_2026_{month}.parquet" for month in ("05","06","07") for side in ("long","short")]

def build_panel(*,representation:Path,failure:Path,label_dir:Path)->pd.DataFrame:
 rep=validate_candidate_identity(pd.read_parquet(representation)); fail=validate_candidate_identity(pd.read_parquet(failure))
 labels=[]
 for p in _label_paths(label_dir):
  if not p.exists(): raise RegimeOOFStackError(f"required alpha label ledger missing: {p}")
  labels.append(pd.read_parquet(p,columns=[*IDENTITY_COLUMNS,ALPHA]))
 alpha=validate_candidate_identity(pd.concat(labels,ignore_index=True))
 panel=rep.merge(fail,on=list(IDENTITY_COLUMNS),how="inner",validate="one_to_one",suffixes=("", "__failure"))
 panel=panel.merge(alpha,on=list(IDENTITY_COLUMNS),how="inner",validate="one_to_one")
 panel=validate_candidate_identity(panel).sort_values(["__ts__","candidate_id"],kind="stable").reset_index(drop=True)
 needed=[TARGET,"execution_gross_ev_12h","execution_cost_return","execution_label_end_utc","catboost__residual__without_hpo__all_features",*sum(feature_lists().values(),[])]
 missing=[c for c in dict.fromkeys(needed) if c not in panel]
 if missing: raise RegimeOOFStackError(f"common panel is missing required fields: {missing}")
 for c in ("execution_label_end_utc",): panel[c]=pd.to_datetime(panel[c],utc=True,errors="raise")
 if panel["execution_label_end_utc"].lt(panel["__ts__"]+LABEL_DELAY).any(): raise RegimeOOFStackError("execution label resolution is earlier than 12h horizon")
 return panel

def _mat(train:pd.DataFrame,ev:pd.DataFrame,cols:list[str])->tuple[pd.DataFrame,pd.DataFrame]:
 a=train[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan); b=ev[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
 med=a.median().fillna(0.0);return a.fillna(med).astype("float32"),b.fillna(med).astype("float32")
def _mapper(x:np.ndarray,y:np.ndarray):
 ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok]
 if len(x)<8 or np.unique(x).size<2:
  v=float(np.mean(y)) if len(y) else 0.;return lambda z:np.full(len(z),v)
 m=IsotonicRegression(out_of_bounds="clip",increasing="auto").fit(x,y);return lambda z:np.asarray(m.predict(np.asarray(z,float)),float)
def _fit(train:pd.DataFrame,ev:pd.DataFrame,cols:list[str],seed:int)->tuple[np.ndarray,np.ndarray,pd.Timestamp]:
 x,z=_mat(train,ev,cols); y=pd.to_numeric(train[TARGET],errors="coerce").fillna(0.).to_numpy(float)
 model=lgb.LGBMRegressor(n_estimators=160,learning_rate=.035,num_leaves=15,min_child_samples=180,subsample=.85,colsample_bytree=.9,reg_lambda=3.,random_state=seed,n_jobs=4,verbosity=-1).fit(x,y)
 raw_train=np.asarray(model.predict(x),float);raw_eval=np.asarray(model.predict(z),float)
 recent_start=pd.to_datetime(train["execution_label_end_utc"],utc=True).max()-pd.Timedelta(days=21)
 recent=train.loc[pd.to_datetime(train["execution_label_end_utc"],utc=True).ge(recent_start)]
 _,xr=_mat(train,recent,cols);mapper=_mapper(np.asarray(model.predict(xr),float),pd.to_numeric(recent[TARGET],errors="coerce").fillna(0.).to_numpy(float))
 return raw_eval,mapper(raw_eval),recent_start
def _rank(a:pd.Series,b:pd.Series)->float:
 a,b=pd.to_numeric(a,errors="coerce"),pd.to_numeric(b,errors="coerce");ok=a.notna()&b.notna();return float(a[ok].rank().corr(b[ok].rank())) if ok.sum()>=3 else float("nan")
def _side(frame:pd.DataFrame,arm:str,cols:EvaluationColumns,top:float)->pd.DataFrame:
 sel=frame.loc[global_top_k_mask(frame,score_col=cols.mapped_score,top_fraction=top)];out=[]
 for side,x in frame.groupby("side_name",observed=True):
  s=sel.loc[sel.side_name.eq(side)];out.append({"arm":arm,"side_name":side,"candidate_rows":len(x),"global_selected_rows":len(s),"alpha_rank_ic":_rank(x[cols.mapped_score],x[cols.alpha_target]),"execution_net_rank_ic":_rank(x[cols.mapped_score],x[cols.net_ev]),"global_top10_net_ev":float(s[cols.net_ev].mean()),"global_top10_hit_rate":float(s[cols.net_ev].gt(0).mean())})
 return pd.DataFrame(out)

def run(*,output:Path=DEFAULT_OUTPUT,representation:Path=DEFAULT_REP,failure:Path=DEFAULT_FAILURE,label_dir:Path=DEFAULT_LABEL_DIR,min_train_rows:int=12000,top_fraction:float=.10)->Path:
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 panel=build_panel(representation=Path(representation),failure=Path(failure),label_dir=Path(label_dir));arms=feature_lists()
 # First two full weeks are train-only.  Thereafter weekly outer blocks are OOF.
 starts=sorted(panel["__ts__"].dt.floor("D").unique());first=pd.to_datetime(starts[0],utc=True); eval_starts=pd.date_range(first+pd.Timedelta(days=14),panel["__ts__"].max().floor("D")+pd.Timedelta(days=1),freq="7D")
 predictions={a:[] for a in arms};diagnostics=[];provenance=[]
 for n,start in enumerate(eval_starts):
  end=start+pd.Timedelta(days=7); ev=panel.loc[(panel.__ts__>=start)&(panel.__ts__<end)].copy(); train=panel.loc[pd.to_datetime(panel.execution_label_end_utc,utc=True).lt(start)].copy()
  if ev.empty or len(train)<min_train_rows:continue
  fid=f"weekly_{start:%Y%m%d}";foldrec={"fold_id":fid,"evaluation_start_utc":start,"evaluation_end_exclusive_utc":end,"train_rows":len(train),"evaluation_rows":len(ev),"train_label_end_max":train.execution_label_end_utc.max()}
  diagnostics.append(ev.loc[:,list(IDENTITY_COLUMNS)+RETAINED_REPRESENTATION_DIAGNOSTICS].assign(unsup_fold_id=fid,representation_train_rows=len(train),representation_train_label_end_max=train.execution_label_end_utc.max(),representation_is_fold_local=False,representation_provenance="precomputed_handoff_only__raw_causal_inputs_not_retained_on_common_surface"))
  for ai,(arm,features) in enumerate(arms.items()):
   parts=[]
   for side,local in ev.groupby("side_name",observed=True):
    tr=train.loc[train.side_name.eq(side)];
    if len(tr)<min_train_rows//3:raise RegimeOOFStackError(f"insufficient side training support {side}")
    raw,mapped,recent_start=_fit(tr,local,features,77+n*13+ai);parts.append(local.loc[:,list(IDENTITY_COLUMNS)].assign(unsup_fold_id=fid,unsup_train_end_utc=start,recent_map_start_utc=recent_start,raw_unsupervised_score=raw,mapped_score=mapped))
   predictions[arm].append(pd.concat(parts,ignore_index=True))
  provenance.append(foldrec)
 if not provenance:raise RegimeOOFStackError("no valid chronological OOF folds")
 output.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(dir=output.parent,prefix=f".{output.name}."))
 try:
  d=temp/'prediction_sidecars';d.mkdir();frames={}
  for arm,parts in predictions.items():
   side=pd.concat(parts,ignore_index=True).sort_values(["__ts__","candidate_id"],kind="stable");validate_candidate_identity(side);side.to_parquet(d/f"{arm}.parquet",index=False);frames[arm]=panel.merge(side,on=list(IDENTITY_COLUMNS),how="inner",validate="one_to_one")
  cols=EvaluationColumns(mapped_score="mapped_score",alpha_target=ALPHA,net_ev=TARGET,gross_ev="execution_gross_ev_12h",cost="execution_cost_return")
  summary,periods,categories=evaluate_matched_arms(frames,columns=cols,top_fraction=top_fraction,category_col=None);sides=[]
  for arm,frame in frames.items():
   sides.append(_side(frame,arm,cols,top_fraction));latest=periods.loc[(periods.arm==arm)&(periods.period_type=="month")].sort_values("period").tail(1);base=summary.loc[summary.arm.eq("baseline")].iloc[0];row=summary.arm.eq(arm)
   summary.loc[row,"latest_month"]=latest.period.iloc[0];summary.loc[row,"latest_month_net_ev"]=latest.mean_net_ev.iloc[0]
   summary.loc[row,"aggregate_incremental_net_ev_vs_baseline"]=summary.loc[row,"top10_mean_net_ev"].iloc[0]-base.top10_mean_net_ev
   summary.loc[row,"latest_incremental_net_ev_vs_baseline"]=latest.mean_net_ev.iloc[0]-periods.loc[(periods.arm=="baseline")&(periods.period_type=="month")&(periods.period==latest.period.iloc[0]),"mean_net_ev"].iloc[0]
   summary.loc[row,"aggregate_and_latest_gate_pass"]=bool(summary.loc[row,"aggregate_incremental_net_ev_vs_baseline"].iloc[0]>0 and summary.loc[row,"latest_incremental_net_ev_vs_baseline"].iloc[0]>0 and summary.loc[row,"top10_mean_net_ev"].iloc[0]>0 and summary.loc[row,"latest_month_net_ev"].iloc[0]>0)
  diag=pd.concat(diagnostics,ignore_index=True);validate_candidate_identity(diag);diag.to_parquet(temp/'representation_diagnostic_sidecar.parquet',index=False);pd.DataFrame(provenance).to_parquet(temp/'fold_provenance.parquet',index=False);summary.to_csv(temp/'metrics_summary.csv',index=False);periods.to_parquet(temp/'period_metrics.parquet',index=False);pd.concat(sides,ignore_index=True).to_parquet(temp/'side_metrics.parquet',index=False);(temp/'feature_lists.json').write_text(json.dumps({"arms":arms,"retained_per_candidate_representation_diagnostics":RETAINED_REPRESENTATION_DIAGNOSTICS,"representation_retention_limitation":"existing common surface stores precomputed handoff embeddings only; fold-local re-fit requires raw causal inputs, which this sealed surface does not retain","regime_layer":"GMM geometry and DAE representations only; posterior, entropy, and compact risk-summary fields excluded","transition_layer":"failure p_failure_destination_3h and p_transition_within_3h are separately ablated and then jointly tested; distinct from regime representation","forbidden_action_tokens":ACTION_TOKENS},indent=2,sort_keys=True)+"\n")
  outs=[temp/x for x in ['fold_provenance.parquet','metrics_summary.csv','period_metrics.parquet','side_metrics.parquet','representation_diagnostic_sidecar.parquet','feature_lists.json']]+sorted(d.glob('*.parquet'))
  manifest={"schema":SCHEMA,"status":"MATCHED_CHRONOLOGICAL_OOF_DIAGNOSTIC_COMPLETE","common_rows_raw":len(panel),"predicted_rows":len(next(iter(frames.values()))),"coverage_limitation":"exact common GMM/DAE plus strict failure-first overlap is May--July 2026 only; no all-era claim","selection":{"basis":"one_pooled_global_top10_after_arm_own_recent_21d_causal_ev_mapping","per_timestamp":False,"per_side":False},"mapping":"each side-local arm map fitted only on its previous resolved trailing-21d model scores/outcomes","promotion_eligible":False,"portfolio_replay":False,"inputs":{str(Path(p).resolve()):sha(Path(p)) for p in [representation,failure,*_label_paths(Path(label_dir))]},"outputs_sha256":{str(p.relative_to(temp)):sha(p) for p in outs}}
  mp=temp/'manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+"\n");(temp/'manifest.sha256').write_text(f"{sha(mp)}  manifest.json\n");os.replace(temp,output);return output
 except Exception:shutil.rmtree(temp,ignore_errors=True);raise
def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--representation',type=Path,default=DEFAULT_REP);p.add_argument('--failure',type=Path,default=DEFAULT_FAILURE);p.add_argument('--label-dir',type=Path,default=DEFAULT_LABEL_DIR);p.add_argument('--min-train-rows',type=int,default=12000);p.add_argument('--top-fraction',type=float,default=.10);return p.parse_args(argv)
if __name__=='__main__':print(run(**vars(parse_args())))
