#!/usr/bin/env python3
"""Bounded historical-only six-class CatBoost FS/HPO/geometry/OOF adapter.

This runner is intentionally independent of the global production seven-class
runner.  It consumes only sealed exact-1m target shards and PIT context shards;
it never accepts realised outcomes as model features.  Selection/HPO use
February rows resolved before March; OOF is March from February and April from
February+resolved March, per side.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.audit_febapr2025_historical_catboost_six_class_gate import CLASS_ORDER, _six  # noqa:E402
from extreme_price_movements.path_archetype_support import PathArchetypeSupportConfig, validate_path_archetype_support  # noqa:E402

SCHEMA="febapr2025_historical_six_class_catboost_v2"
IDENTITY=["candidate_id","side_name","__symbol__","__ts__"]
FORBIDDEN=("path_arch_","path_archetype","mfe","mae","future","outcome","target","label","execution_","exit_","__w__","__first_touch")

def _sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _write(p:Path,x:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".partial");t.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+"\n");t.replace(p)
def _write_parquet(p:Path,x:pd.DataFrame)->None:
 """Atomic, resumable parquet output for each independently valid OOF month."""
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+".partial")
 x.to_parquet(t,index=False,compression="zstd")
 t.replace(p)
def _read(p:Path)->Any:return json.loads(p.read_text())
def _guard()->None:
 try:
  import psutil
  if psutil.virtual_memory().available < 2*1024**3: raise RuntimeError("resource guard: <2GiB RAM available")
 except ImportError: pass
def _cat():
 from catboost import CatBoostClassifier
 return CatBoostClassifier

def _labels(root:Path)->pd.DataFrame:
 ix=_read(root/"index.json")
 if not ix.get("coverage",{}).get("complete"):raise ValueError("sealed labels incomplete")
 cols=[*IDENTITY,"__label_end_ts__","path_shape_archetype","path_arch_peak_mfe_r","path_arch_final_return_r"]
 fs=[]
 for s in ix["shards"]:
  p=Path(s["labels"])
  if _sha(p)!=s["sha256"]:raise ValueError(f"label digest mismatch {p}")
  fs.append(pd.read_parquet(p,columns=cols))
 x=pd.concat(fs,ignore_index=True);x["class_label"]=_six(x.path_shape_archetype).astype(str)
 return x
def _context(root:Path,side:str)->pd.DataFrame:
 # Each context shard is immutable per feature-store symbol.  Loading one at a
 # time bounds peak read memory and preserves the sealed PIT schema.
 parts=[]
 for p in sorted((root/"shards").glob("*.parquet")):
  q=pd.read_parquet(p);q=q.loc[q.side_name.astype(str).eq(side)]
  if len(q):parts.append(q)
  _guard()
 return pd.concat(parts,ignore_index=True)
def _features(frame:pd.DataFrame)->list[str]:
 excluded=set(IDENTITY+["__decision_ts__","class_label","path_shape_archetype","__label_end_ts__"])
 cols=[]
 for c in frame.columns:
  if c in excluded:continue
  low=c.lower()
  if any(token in low for token in FORBIDDEN):continue
  if pd.api.types.is_numeric_dtype(frame[c]):cols.append(c)
 if not cols:raise ValueError("no numeric pre-entry features")
 return cols
def _frame(args:argparse.Namespace,side:str)->pd.DataFrame:
 labels=_labels(args.labels_root);ctx=_context(args.context_root,side)
 # Context uses the feature-store underscore symbol spelling, whereas sealed
 # execution labels retain the candidate-ID slash spelling. Candidate ID/time/
 # side are the immutable authority; validate the reversible spelling after.
 lab=labels.loc[labels.side_name.eq(side)].copy()
 x=ctx.merge(lab.drop(columns="__symbol__"),on=["candidate_id","side_name","__ts__"],how="inner",validate="one_to_one")
 if len(x)!=102597:raise ValueError(f"side {side} exact PIT/label join expected 102597, got {len(x)}")
 x["__ts__"]=pd.to_datetime(x["__ts__"],utc=True);x["__label_end_ts__"]=pd.to_datetime(x["__label_end_ts__"],utc=True)
 if not (x["__label_end_ts__"]==x["__ts__"]+pd.Timedelta(hours=13)).all():
  # Signal timestamp +1h decision, then exact 12h path resolution.
  raise ValueError("label timing must be signal + 13h")
 return x.sort_values("__ts__",kind="stable").reset_index(drop=True)
def _params(trial:int=0,iterations:int=600)->dict[str,Any]:
 grid=[(5,.04,12),(6,.03,20),(7,.025,30),(5,.06,30),(6,.05,12),(7,.035,50)]
 d,lr,l2=grid[trial%len(grid)]
 return {"loss_function":"MultiClass","eval_metric":"MultiClass","iterations":int(iterations),"od_wait":max(20,min(60,int(iterations)//4)),"learning_rate":lr,"depth":d,"l2_leaf_reg":l2,"random_seed":20260729+trial,"verbose":False,"allow_writing_files":False,"thread_count":1,"auto_class_weights":"Balanced"}
def _fit(X:pd.DataFrame,y:pd.Series,Xv:pd.DataFrame,yv:pd.Series,params:dict[str,Any]):
 M=_cat();m=M(**params);m.fit(X,y,eval_set=(Xv,yv),early_stopping_rounds=params["od_wait"],verbose=False);return m
def _refit(X:pd.DataFrame,y:pd.Series,params:dict[str,Any],iterations:int):
 M=_cat();q=dict(params);q["iterations"]=max(1,int(iterations));m=M(**q);m.fit(X,y,verbose=False);return m
def _geometry_labels(x:pd.DataFrame,threshold:float)->pd.Series:
 y=x.class_label.astype(str).copy();raw=x.path_shape_archetype.astype(str)
 affected=raw.isin(("early_mfe_full_reversal","noisy_timeout_usable_mfe"))
 y.loc[affected & x.path_arch_peak_mfe_r.lt(float(threshold))]="dead_timeout"
 return y
def _convergence(trials:list[dict[str,Any]],iterations:int)->dict[str,Any]:
 """Accept only a winner whose early-stopping optimum is meaningfully pre-cap."""
 margin=max(5,int(iterations)//20)
 eligible=[q for q in trials if int(q["best_iteration"])<=int(iterations)-margin-1]
 raw=min(trials,key=lambda q:q["validation_multiclass_logloss"])
 return {"accepted":bool(eligible),"iteration_cap":int(iterations),"required_margin":margin,
         "raw_winner":raw,"eligible_trials":[int(q["trial"]) for q in eligible],
         "reason":"winner selected only from pre-cap early-stopping trials" if eligible else "all trials reach the iteration cap; extend HPO before OOF"}
def _feb_masks(x:pd.DataFrame)->tuple[pd.Series,pd.Series]:
 feb=x.__ts__.lt(pd.Timestamp("2025-03-01",tz="UTC"));resolved=x.__label_end_ts__.lt(pd.Timestamp("2025-03-01",tz="UTC"));return feb&resolved,feb&resolved
def selection(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side; ck=out/"selection.json"
 if ck.exists():return _read(ck)
 x=_frame(args,side);f=_features(x);train,_=_feb_masks(x);z=x.loc[train];cut=pd.Timestamp("2025-02-22",tz="UTC");tr=z.__ts__.lt(cut);va=~tr
 # Two chronological inner folds yield a train-only stability screen; this is
 # deliberately not called MDA.  Both validation blocks are inside February.
 cuts=[pd.Timestamp("2025-02-16",tz="UTC"),pd.Timestamp("2025-02-22",tz="UTC")];ranks=[];folds=[]
 for cut_i in cuts:
  t=z.__ts__.lt(cut_i);v=(z.__ts__>=cut_i)&(z.__ts__<cut_i+pd.Timedelta(days=6))
  m=_fit(z.loc[t,f],z.loc[t,"class_label"],z.loc[v,f],z.loc[v,"class_label"],_params(iterations=args.selection_iterations))
  ranks.append(pd.Series(m.get_feature_importance(),index=f).rank(ascending=False,method="average"));folds.append((m,z.loc[v,f].copy(),z.loc[v,"class_label"].copy()))
 avg=pd.concat(ranks,axis=1).mean(axis=1).sort_values();picked=avg.head(min(48,len(avg))).index.tolist();mda={}
 # Validation-only permutation stability, with models fitted solely on each
 # preceding February fold.  This is MDA evidence, not built-in importance.
 for col in picked:
  losses=[]
  for k,(m,Xv,yv) in enumerate(folds):
   classes=list(m.classes_);yi=pd.Categorical(yv,categories=classes).codes;base=-np.log(np.clip(m.predict_proba(Xv)[np.arange(len(yi)),yi],1e-12,1)).mean();xp=Xv.copy();xp[col]=np.random.default_rng(20260729+k).permutation(xp[col].to_numpy());loss=-np.log(np.clip(m.predict_proba(xp)[np.arange(len(yi)),yi],1e-12,1)).mean();losses.append(float(loss-base))
  mda[col]={"fold_loss_deltas":losses,"mean_loss_delta":float(np.mean(losses))}
 picked=sorted(picked,key=lambda c:mda[c]["mean_loss_delta"],reverse=True)
 r={"schema":SCHEMA,"stage":"selection","method":"two_fold_chronological_importance_plus_validation_permutation_stability","side":side,"class_order":list(CLASS_ORDER),"features":picked,"mean_importance_rank":avg.head(80).to_dict(),"permutation_mda":mda,"train_rows":int(tr.sum()),"validation_rows":int(va.sum()),"training_window":"February resolved before March only","forbidden_outcomes":True}
 _write(ck,r);return r
def geometry(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side;ck=out/"geometry.json"
 if ck.exists():return _read(ck)
 s=selection(args,side)
 x=_frame(args,side);train,_=_feb_masks(x);z=x.loc[train];cut=pd.Timestamp("2025-02-22",tz="UTC");tr=z.__ts__.lt(cut);va=~tr;candidates=[]
 for threshold in (1.5,2.0,3.0):
  yt=_geometry_labels(z,float(threshold));support=validate_path_archetype_support(pd.DataFrame({"path_geometry_label":yt,"__ts__":z.__ts__,"side_name":z.side_name}),PathArchetypeSupportConfig(label_column="path_geometry_label",timestamp_column="__ts__",side_column="side_name",classes=CLASS_ORDER,min_global_class_share=.01,min_month_side_class_share=.005));item={"peak_mfe_r_threshold":threshold,"predeclared_rule_variant":True,"support_passed":support.accepted}
  if support.accepted:
   m=_fit(z.loc[tr,s["features"]],yt.loc[tr],z.loc[va,s["features"]],yt.loc[va],_params(iterations=args.selection_iterations));p=m.predict_proba(z.loc[va,s["features"]]);classes=list(m.classes_);yi=pd.Categorical(yt.loc[va],categories=classes).codes;item["inner_logloss"]=float(-np.log(np.clip(p[np.arange(len(yi)),yi],1e-12,1)).mean())
  else:item["violations"]=support.violations.to_dict(orient="records")
  candidates.append(item)
 eligible=[q for q in candidates if q["support_passed"]]
 if not eligible:raise ValueError("no discriminating geometry variant passes side support")
 winner=min(eligible,key=lambda q:q["inner_logloss"])
 r={"schema":SCHEMA,"stage":"geometry","side":side,"class_order":list(CLASS_ORDER),"selected_features":s["features"],"candidates":candidates,"winner":winner,"selection_basis":"February inner multiclass logloss only; never held-out EV"}
 _write(ck,r);return r
def hpo(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side;ck=out/"hpo.json";prog=out/"hpo_progress.json"
 if ck.exists():
  prior=_read(ck)
  if int(prior.get("hpo_iterations",-1))!=int(args.hpo_iterations):raise ValueError("HPO iteration contract differs; use a fresh output root")
  if len(prior.get("trials",[]))>=int(args.hpo_trials):return prior
 g=geometry(args,side);x=_frame(args,side);train,_=_feb_masks(x);z=x.loc[train].copy();z["class_label"]=_geometry_labels(z,float(g["winner"]["peak_mfe_r_threshold"]));cut=pd.Timestamp("2025-02-22",tz="UTC");tr=z.__ts__.lt(cut);va=~tr;done=_read(prog) if prog.exists() else {"trials":[]}
 done_ids={int(q["trial"]) for q in done["trials"]}
 for i in range(int(args.hpo_trials)):
  if i in done_ids:continue
  params=_params(i,args.hpo_iterations);m=_fit(z.loc[tr,g["selected_features"]],z.loc[tr,"class_label"],z.loc[va,g["selected_features"]],z.loc[va,"class_label"],params);p=m.predict_proba(z.loc[va,g["selected_features"]]);classes=list(m.classes_);yi=pd.Categorical(z.loc[va,"class_label"],categories=classes).codes
  loss=float(-np.log(np.clip(p[np.arange(len(yi)),yi],1e-12,1)).mean());done["trials"].append({"trial":i,"params":params,"best_iteration":int(m.get_best_iteration()),"validation_multiclass_logloss":loss});_write(prog,done);_guard()
 conv=_convergence(done["trials"],int(args.hpo_iterations));eligible=[q for q in done["trials"] if int(q["trial"]) in set(conv["eligible_trials"])]
 best=min(eligible,key=lambda q:q["validation_multiclass_logloss"]) if eligible else None
 r={"schema":SCHEMA,"stage":"hpo","side":side,"class_order":list(CLASS_ORDER),"selected_features":g["selected_features"],"winner":best,"convergence":conv,"trials":done["trials"],"hpo_iterations":int(args.hpo_iterations),"training_window":"February resolved before March only"};_write(ck,r);return r
def oof(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side;ck=out/"oof_manifest.json";op=out/"oof.parquet"
 if ck.exists() and op.exists():return _read(ck)
 h=hpo(args,side)
 if not h.get("convergence",{}).get("accepted") or h.get("winner") is None:raise ValueError("HPO has not converged; strict OOF is forbidden")
 g=geometry(args,side);x=_frame(args,side);x["class_label"]=_geometry_labels(x,float(g["winner"]["peak_mfe_r_threshold"]));f=h["selected_features"];p=h["winner"]["params"];records=[]
 for month,start,end in (("2025-03",pd.Timestamp("2025-03-01",tz="UTC"),pd.Timestamp("2025-04-01",tz="UTC")),("2025-04",pd.Timestamp("2025-04-01",tz="UTC"),pd.Timestamp("2025-05-01",tz="UTC"))):
  month_path=out/f"oof_{month}.parquet"
  if month_path.exists():
   q=pd.read_parquet(month_path)
   required={*IDENTITY,"__label_end_ts__","class_label","oof_month","train_rows","inner_train_rows","inner_validation_rows","inner_best_iteration","latest_train_label_end_utc","validation_start_utc",*(f"prob_{c}" for c in CLASS_ORDER)}
   if not required.issubset(q.columns) or not q.oof_month.astype(str).eq(month).all():raise ValueError(f"invalid OOF checkpoint {month_path}")
   records.append(q);continue
  test=x.loc[(x.__ts__>=start)&(x.__ts__<end)].copy()
  fit=x.loc[(x.__ts__<start)&(x.__label_end_ts__<start)].copy()
  if month=="2025-03": fit=fit.loc[fit.__ts__.lt(pd.Timestamp("2025-03-01",tz="UTC"))]
  if fit.empty or test.empty:raise ValueError(f"missing strict {month} fit/test cohort")
  # Outer March/April labels are never passed to fitting/early stopping.  The
  # best iteration is determined on a chronological inner tail of prior,
  # already-resolved rows, then the model is refit on all prior rows.
  inner_cut=fit.__ts__.quantile(0.80);inner_train=fit.loc[fit.__ts__<inner_cut];inner_val=fit.loc[fit.__ts__>=inner_cut]
  inner=_fit(inner_train[f],inner_train.class_label,inner_val[f],inner_val.class_label,p)
  best=max(1,int(inner.get_best_iteration())+1);m=_refit(fit[f],fit.class_label,p,best)
  proba=m.predict_proba(test[f]);classes=list(m.classes_)
  aligned=np.zeros((len(test),len(CLASS_ORDER)),dtype=np.float32)
  for j,c in enumerate(classes):aligned[:,CLASS_ORDER.index(c)]=proba[:,j]
  q=test[IDENTITY+["__label_end_ts__","class_label"]].copy();q["oof_month"]=month;q["train_rows"]=len(fit);q["inner_train_rows"]=len(inner_train);q["inner_validation_rows"]=len(inner_val);q["inner_best_iteration"]=best;q["latest_train_label_end_utc"]=fit.__label_end_ts__.max();q["validation_start_utc"]=start
  q["predicted_class"]=[CLASS_ORDER[i] for i in aligned.argmax(axis=1)]
  for j,c in enumerate(CLASS_ORDER):q[f"prob_{c}"]=aligned[:,j]
  _write_parquet(month_path,q)
  records.append(q)
  _guard()
 o=pd.concat(records,ignore_index=True)
 # Outcome is joined only after probabilities are frozen, for an OOF report.
 econ=pd.read_parquet(args.population,columns=["candidate_id","execution_net_ev_12h"])
 o=o.merge(econ,on="candidate_id",validate="one_to_one")
 o.rename(columns={"execution_net_ev_12h":"realized_execution_net_ev_12h"},inplace=True)
 _write_parquet(op,o)
 report=o.groupby("predicted_class").realized_execution_net_ev_12h.agg(["count","mean","median"]).reset_index().to_dict(orient="records")
 r={"schema":SCHEMA,"stage":"strict_oof","side":side,"class_order":list(CLASS_ORDER),"oof":str(op),"rows":int(len(o)),"strict_provenance":"March train=February rows with label_end<Mar1; April train=all rows with label_end<Apr1","hpo_convergence":h["convergence"],"economics_audit_only":report,"outcome_not_model_input":True,"twelve_hour_derivative_not_frozen_24h_v6":True}
 _write(ck,r);return r
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--labels-root",type=Path,required=True);p.add_argument("--context-root",type=Path,required=True);p.add_argument("--population",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--side",choices=("long","short"),required=True);p.add_argument("--stage",choices=("selection","geometry","hpo","oof"),default="oof");p.add_argument("--hpo-trials",type=int,default=6);p.add_argument("--selection-iterations",type=int,default=128);p.add_argument("--hpo-iterations",type=int,default=256);a=p.parse_args()
 out={"selection":selection,"geometry":geometry,"hpo":hpo,"oof":oof}[a.stage](a,a.side);print(json.dumps(out,sort_keys=True,default=str))
if __name__=="__main__":main()
