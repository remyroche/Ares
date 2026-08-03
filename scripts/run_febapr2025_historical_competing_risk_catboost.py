#!/usr/bin/env python3
"""Separate 3-class soft-triple-barrier competing-risk CatBoost challenger."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_febapr2025_historical_six_class_catboost import (IDENTITY,_context,_convergence,_features,_guard,_params,_sha,_write,_write_parquet) # noqa:E402

SCHEMA="febapr2025_historical_competing_risk_catboost_v1"
RISK_ORDER=("timeout","adverse_first_or_conflict","favorable_first")

def _cat():
 from catboost import CatBoostClassifier
 return CatBoostClassifier
def _labels(root:Path)->pd.DataFrame:
 ix=json.loads((root/"index.json").read_text())
 if not ix.get("coverage",{}).get("complete"):raise ValueError("sealed labels incomplete")
 cols=[*IDENTITY,"__label_end_ts__","__soft_tb_first_event__","__soft_tb_order_ambiguous__","__soft_tb_upper_hit_12h__","__soft_tb_lower_hit_12h__"]
 out=[]
 for s in ix["shards"]:
  p=Path(s["labels"])
  if _sha(p)!=s["sha256"]:raise ValueError(f"label digest mismatch {p}")
  out.append(pd.read_parquet(p,columns=cols))
 x=pd.concat(out,ignore_index=True);x["risk_class"]=x["__soft_tb_first_event__"].astype(str)
 if not x.risk_class.isin(RISK_ORDER).all():raise ValueError("invalid soft triple-barrier competing-risk class")
 return x
def _frame(args:argparse.Namespace,side:str)->pd.DataFrame:
 labels=_labels(args.labels_root);ctx=_context(args.context_root,side);lab=labels.loc[labels.side_name.eq(side)].copy()
 x=ctx.merge(lab.drop(columns="__symbol__"),on=["candidate_id","side_name","__ts__"],how="inner",validate="one_to_one")
 if len(x)!=102597:raise ValueError(f"side {side} exact PIT/label join expected 102597, got {len(x)}")
 x["__ts__"]=pd.to_datetime(x["__ts__"],utc=True);x["__label_end_ts__"]=pd.to_datetime(x["__label_end_ts__"],utc=True)
 if not (x["__label_end_ts__"]==x["__ts__"]+pd.Timedelta(hours=13)).all():raise ValueError("label timing must be signal + 13h")
 return x.sort_values("__ts__",kind="stable").reset_index(drop=True)
def _weight(x:pd.DataFrame)->np.ndarray:
 # Recorded intrabar-order ambiguity remains adverse but receives reduced
 # influence, preserving conservative class semantics without inventing order.
 return np.where(x["__soft_tb_order_ambiguous__"].astype(bool).to_numpy(),.35,1.).astype(np.float32)
def _pit_features(x:pd.DataFrame)->list[str]:
 """The stored hit/order columns are labels, never candidate-time inputs."""
 return [c for c in _features(x) if not c.startswith("__soft_tb_")]
def _fit(X:pd.DataFrame,y:pd.Series,Xv:pd.DataFrame,yv:pd.Series,p:dict[str,Any],w:np.ndarray):
 m=_cat()(**p);m.fit(X,y,sample_weight=w,eval_set=(Xv,yv),early_stopping_rounds=p["od_wait"],verbose=False);return m
def _refit(X:pd.DataFrame,y:pd.Series,p:dict[str,Any],iterations:int,w:np.ndarray):
 q=dict(p);q["iterations"]=max(1,int(iterations));m=_cat()(**q);m.fit(X,y,sample_weight=w,verbose=False);return m
def _feb(x:pd.DataFrame)->pd.Series:
 return x.__ts__.lt(pd.Timestamp("2025-03-01",tz="UTC"))&x.__label_end_ts__.lt(pd.Timestamp("2025-03-01",tz="UTC"))
def _loss(m:Any,X:pd.DataFrame,y:pd.Series)->float:
 p=m.predict_proba(X);cs=list(m.classes_);yi=pd.Categorical(y,categories=cs).codes
 return float(-np.log(np.clip(p[np.arange(len(yi)),yi],1e-12,1)).mean())
def selection(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side;ck=out/"selection.json"
 if ck.exists():return json.loads(ck.read_text())
 x=_frame(args,side);f=_pit_features(x)
 if not f:raise ValueError("no PIT features after excluding triple-barrier outcome fields")
 z=x.loc[_feb(x)].copy();folds=[];ranks=[]
 for cut in (pd.Timestamp("2025-02-16",tz="UTC"),pd.Timestamp("2025-02-22",tz="UTC")):
  t=z.__ts__.lt(cut);v=(z.__ts__>=cut)&(z.__ts__<cut+pd.Timedelta(days=6))
  m=_fit(z.loc[t,f],z.loc[t,"risk_class"],z.loc[v,f],z.loc[v,"risk_class"],_params(iterations=args.selection_iterations),_weight(z.loc[t]))
  ranks.append(pd.Series(m.get_feature_importance(),index=f).rank(ascending=False));folds.append((m,z.loc[v,f].copy(),z.loc[v,"risk_class"].copy()))
 avg=pd.concat(ranks,axis=1).mean(axis=1).sort_values();picked=avg.head(min(48,len(avg))).index.tolist();mda={}
 for col in picked:
  deltas=[]
  for k,(m,Xv,yv) in enumerate(folds):
   base=_loss(m,Xv,yv);q=Xv.copy();q[col]=np.random.default_rng(20260730+k).permutation(q[col].to_numpy());deltas.append(_loss(m,q,yv)-base)
  mda[col]={"fold_loss_deltas":[float(v) for v in deltas],"mean_loss_delta":float(np.mean(deltas))}
 picked=sorted(picked,key=lambda c:mda[c]["mean_loss_delta"],reverse=True)
 r={"schema":SCHEMA,"stage":"selection","side":side,"risk_order":list(RISK_ORDER),"method":"two_fold_chronological_importance_plus_validation_permutation_stability","features":picked,"permutation_mda":mda,"training_window":"February resolved before March only","forbidden_outcomes":True}
 _write(ck,r);return r
def hpo(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side;ck=out/"hpo.json";prog=out/"hpo_progress.json"
 if ck.exists():
  prior=json.loads(ck.read_text())
  if int(prior.get("hpo_iterations",-1))!=args.hpo_iterations:raise ValueError("HPO iteration contract differs; use a fresh output root")
  if len(prior.get("trials",[]))>=args.hpo_trials:return prior
 s=selection(args,side);x=_frame(args,side);z=x.loc[_feb(x)].copy();cut=pd.Timestamp("2025-02-22",tz="UTC");t=z.__ts__.lt(cut);v=~t;done=json.loads(prog.read_text()) if prog.exists() else {"trials":[]};ids={int(q["trial"]) for q in done["trials"]}
 for i in range(args.hpo_trials):
  if i in ids:continue
  p=_params(i,args.hpo_iterations);m=_fit(z.loc[t,s["features"]],z.loc[t,"risk_class"],z.loc[v,s["features"]],z.loc[v,"risk_class"],p,_weight(z.loc[t]));done["trials"].append({"trial":i,"params":p,"best_iteration":int(m.get_best_iteration()),"validation_multiclass_logloss":_loss(m,z.loc[v,s["features"]],z.loc[v,"risk_class"])});_write(prog,done);_guard()
 conv=_convergence(done["trials"],args.hpo_iterations);eligible=[q for q in done["trials"] if q["trial"] in set(conv["eligible_trials"])];winner=min(eligible,key=lambda q:q["validation_multiclass_logloss"]) if eligible else None
 r={"schema":SCHEMA,"stage":"hpo","side":side,"risk_order":list(RISK_ORDER),"selected_features":s["features"],"trials":done["trials"],"winner":winner,"convergence":conv,"hpo_iterations":args.hpo_iterations,"ambiguity_training_weight":.35,"training_window":"February resolved before March only"};_write(ck,r);return r
def oof(args:argparse.Namespace,side:str)->dict[str,Any]:
 out=args.output_dir/side;ck=out/"oof_manifest.json";op=out/"oof.parquet"
 if ck.exists() and op.exists():return json.loads(ck.read_text())
 h=hpo(args,side)
 if not h["convergence"]["accepted"] or h["winner"] is None:raise ValueError("HPO has not converged; strict OOF forbidden")
 x=_frame(args,side);f=h["selected_features"];p=h["winner"]["params"];records=[]
 for month,start,end in (("2025-03",pd.Timestamp("2025-03-01",tz="UTC"),pd.Timestamp("2025-04-01",tz="UTC")),("2025-04",pd.Timestamp("2025-04-01",tz="UTC"),pd.Timestamp("2025-05-01",tz="UTC"))):
  mp=out/f"oof_{month}.parquet"
  if mp.exists():records.append(pd.read_parquet(mp));continue
  test=x.loc[(x.__ts__>=start)&(x.__ts__<end)].copy();fit=x.loc[(x.__ts__<start)&(x.__label_end_ts__<start)].copy();cut=fit.__ts__.quantile(.8);inner_train=fit.loc[fit.__ts__<cut];inner_val=fit.loc[fit.__ts__>=cut]
  inner=_fit(inner_train[f],inner_train.risk_class,inner_val[f],inner_val.risk_class,p,_weight(inner_train));best=max(1,int(inner.get_best_iteration())+1);m=_refit(fit[f],fit.risk_class,p,best,_weight(fit));proba=m.predict_proba(test[f]);aligned=np.zeros((len(test),len(RISK_ORDER)),dtype=np.float32)
  for j,c in enumerate(m.classes_):aligned[:,RISK_ORDER.index(c)]=proba[:,j]
  q=test[IDENTITY+["__label_end_ts__","risk_class","__soft_tb_order_ambiguous__"]].copy();q["oof_month"]=month;q["train_rows"]=len(fit);q["inner_train_rows"]=len(inner_train);q["inner_validation_rows"]=len(inner_val);q["inner_best_iteration"]=best;q["latest_train_label_end_utc"]=fit.__label_end_ts__.max();q["predicted_class"]=[RISK_ORDER[j] for j in aligned.argmax(axis=1)]
  for j,c in enumerate(RISK_ORDER):q[f"prob_{c}"]=aligned[:,j]
  _write_parquet(mp,q);records.append(q);_guard()
 o=pd.concat(records,ignore_index=True);econ=pd.read_parquet(args.population,columns=["candidate_id","execution_net_ev_12h"]);o=o.merge(econ,on="candidate_id",validate="one_to_one");o.rename(columns={"execution_net_ev_12h":"realized_execution_net_ev_12h"},inplace=True);_write_parquet(op,o)
 r={"schema":SCHEMA,"stage":"strict_oof","side":side,"risk_order":list(RISK_ORDER),"rows":len(o),"oof":str(op),"strict_provenance":"March train=February resolved labels; April train=all labels resolved before April","hpo_convergence":h["convergence"],"soft_triple_barrier_contract":"upper=max(1.5 ATR,1.5%); lower=1.0 ATR; timeout=12h; conflict=adverse; recorded ambiguity weighted 0.35","outcome_not_model_input":True};_write(ck,r);return r
def _metrics(x:pd.DataFrame)->dict[str,Any]:
 p=x[[f"prob_{c}" for c in RISK_ORDER]].to_numpy(float);y=pd.Categorical(x.risk_class,categories=RISK_ORDER).codes;pred=p.argmax(1);one=np.eye(3)[y];conf=p.max(1);bins=np.minimum((conf*10).astype(int),9);ece=sum((bins==b).mean()*abs(conf[bins==b].mean()-(pred[bins==b]==y[bins==b]).mean()) for b in range(10) if (bins==b).any());classes=[]
 for j,c in enumerate(RISK_ORDER):
  z=(y==j).astype(int);classes.append({"class":c,"actual_share":float(z.mean()),"mean_probability":float(p[:,j].mean()),"roc_auc":float(roc_auc_score(z,p[:,j])),"average_precision":float(average_precision_score(z,p[:,j]))})
 return {"rows":len(x),"logloss":float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),"accuracy":float((pred==y).mean()),"macro_f1":float(f1_score(y,pred,average="macro")),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"brier":float(np.square(p-one).sum(1).mean()),"top_confidence_ece_10bin":float(ece),"class_calibration_and_discrimination":classes}
def _top(x:pd.DataFrame,score:str)->dict[str,Any]:
 q=x.sort_values(score,ascending=False,kind="stable").head(int(np.ceil(len(x)*.1)));v=q.realized_execution_net_ev_12h
 return {"rows":len(q),"threshold":float(q[score].iloc[-1]),"mean_net_ev":float(v.mean()),"median_net_ev":float(v.median()),"positive_ev_rate":float(v.gt(0).mean())}
def report(args:argparse.Namespace)->dict[str,Any]:
 if args.six_class_root is None:raise ValueError("--six-class-root required for identical-row comparison")
 rows=[]
 for side in ("long","short"):
  m=json.loads((args.output_dir/side/"oof_manifest.json").read_text())
  if not m.get("hpo_convergence",{}).get("accepted"):raise ValueError("unconverged challenger OOF")
  rows.append(pd.read_parquet(args.output_dir/side/"oof.parquet").assign(side=side))
 x=pd.concat(rows,ignore_index=True);per=[]
 for (side,month),q in x.groupby(["side","oof_month"],sort=True):per.append({"side":side,"month":month,"metrics":_metrics(q),"raw_global_top10_favorable_probability":_top(q,"prob_favorable_first")})
 six=pd.concat([pd.read_parquet(args.six_class_root/s/"oof.parquet") for s in ("long","short")],ignore_index=True);six["six_raw_actionable_probability"]=six.prob_fast_realization_winner+six.prob_late_breakout+six.prob_slow_grinder
 z=x.merge(six[["candidate_id","side_name","__ts__","six_raw_actionable_probability"]],on=["candidate_id","side_name","__ts__"],validate="one_to_one")
 r={"schema":SCHEMA+"_report","scope":"separate competing-risk challenger; never merged with six-class architecture","overall":_metrics(x),"by_side_month":per,"raw_global_top10_favorable_probability":_top(x,"prob_favorable_first"),"identical_row_six_class_comparison":{"rows":len(z),"probability_correlation":float(z.prob_favorable_first.corr(z.six_raw_actionable_probability)),"competing_risk_raw_global_top10":_top(z,"prob_favorable_first"),"six_class_raw_global_top10":_top(z,"six_raw_actionable_probability")},"not_a_deployed_policy":True};_write(args.output_dir/"strict_oof_metrics_calibration_economics_comparison.json",r);return r
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--labels-root",type=Path,required=True);p.add_argument("--context-root",type=Path,required=True);p.add_argument("--population",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--six-class-root",type=Path);p.add_argument("--side",choices=("long","short"));p.add_argument("--stage",choices=("selection","hpo","oof","report"),required=True);p.add_argument("--selection-iterations",type=int,default=32);p.add_argument("--hpo-iterations",type=int,default=128);p.add_argument("--hpo-trials",type=int,default=6);a=p.parse_args();fn={"selection":selection,"hpo":hpo,"oof":oof}[a.stage] if a.stage!="report" else lambda _:report(a)
 if a.stage!="report" and a.side is None:p.error("--side is required unless --stage report")
 print(json.dumps(fn(a,a.side) if a.stage!="report" else fn(a),sort_keys=True,default=str))
if __name__=="__main__":main()
