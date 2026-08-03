#!/usr/bin/env python3
"""Causal calibration/context audit for separate six-class and competing-risk OOF streams."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_febapr2025_historical_six_class_catboost import (CLASS_ORDER,IDENTITY,_frame as six_frame,_geometry_labels,_refit as six_refit,_write,_write_parquet) # noqa:E402
from scripts.run_febapr2025_historical_competing_risk_catboost import (RISK_ORDER,_frame as risk_frame,_refit as risk_refit,_weight) # noqa:E402

SCHEMA="historical_path_head_causal_calibration_context_v1"
SIX_ACTION=("fast_realization_winner","late_breakout","slow_grinder")

def _softmax(logp:np.ndarray,t:float)->np.ndarray:
 z=logp/max(float(t),1e-3);z-=z.max(axis=1,keepdims=True);e=np.exp(z);return e/e.sum(axis=1,keepdims=True)
def _temperature(p:np.ndarray,y:np.ndarray)->float:
 logp=np.log(np.clip(p,1e-8,1.))
 grid=np.linspace(.35,4.,147);loss=[-np.log(np.clip(_softmax(logp,t)[np.arange(len(y)),y],1e-12,1.)).mean() for t in grid]
 return float(grid[int(np.argmin(loss))])
def _metrics(x:pd.DataFrame,classes:tuple[str,...],prefix:str)->dict[str,Any]:
 p=x[[f"{prefix}{c}" for c in classes]].to_numpy(float);y=pd.Categorical(x["label"],categories=classes).codes;pred=p.argmax(1);one=np.eye(len(classes))[y];conf=p.max(1);bins=np.minimum((conf*10).astype(int),9);ece=sum((bins==b).mean()*abs(conf[bins==b].mean()-(pred[bins==b]==y[bins==b]).mean()) for b in range(10) if (bins==b).any());pc=[]
 for j,c in enumerate(classes):
  z=(y==j).astype(int);pc.append({"class":c,"actual_share":float(z.mean()),"mean_probability":float(p[:,j].mean()),"signed_gap":float(p[:,j].mean()-z.mean()),"roc_auc":float(roc_auc_score(z,p[:,j])),"average_precision":float(average_precision_score(z,p[:,j]))})
 return {"rows":len(x),"logloss":float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1.)).mean()),"accuracy":float((pred==y).mean()),"macro_f1":float(f1_score(y,pred,average="macro")),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"brier":float(np.square(p-one).sum(1).mean()),"top_confidence_ece_10bin":float(ece),"per_class":pc}
def _top(x:pd.DataFrame,score:str)->dict[str,Any]:
 q=x.sort_values(score,ascending=False,kind="stable").head(int(np.ceil(len(x)*.1)));v=q.realized_execution_net_ev_12h
 return {"rows":len(q),"threshold":float(q[score].iloc[-1]),"mean_net_ev":float(v.mean()),"median_net_ev":float(v.median()),"positive_ev_rate":float(v.gt(0).mean())}
def _jaccard_top(x:pd.DataFrame,left:str,right:str)->float:
 n=int(np.ceil(len(x)*.1));a=set(x.nlargest(n,left).candidate_id);b=set(x.nlargest(n,right).candidate_id)
 return float(len(a&b)/len(a|b))
def _load_hpo(root:Path,side:str)->dict[str,Any]:
 h=json.loads((root/side/"hpo.json").read_text())
 if not h.get("convergence",{}).get("accepted") or h.get("winner") is None:raise ValueError(f"unconverged {root}/{side}")
 return h
def _predict_inner(frame:pd.DataFrame,side:str,family:str,hpo:dict[str,Any],start:pd.Timestamp)->pd.DataFrame:
 prior=frame.loc[(frame.__ts__<start)&(frame.__label_end_ts__<start)].copy();features=hpo["selected_features"];params=hpo["winner"]["params"];iterations=int(hpo["winner"]["best_iteration"])+1;classes=CLASS_ORDER if family=="six" else RISK_ORDER;label="class_label" if family=="six" else "risk_class";parts=[]
 cuts=[prior.__ts__.quantile(.50),prior.__ts__.quantile(.75)]
 for i,cut in enumerate(cuts):
  train=prior.loc[prior.__ts__<cut];val=prior.loc[(prior.__ts__>=cut)&(prior.__ts__<(cuts[1] if i==0 else start))]
  if len(train)==0 or len(val)==0:raise ValueError("empty prior inner OOF split")
  if family=="six":m=six_refit(train[features],train[label],params,iterations)
  else:m=risk_refit(train[features],train[label],params,iterations,_weight(train))
  p=m.predict_proba(val[features]);a=np.zeros((len(val),len(classes)),dtype=np.float32)
  for j,c in enumerate(m.classes_):a[:,classes.index(c)]=p[:,j]
  q=val[IDENTITY+["__label_end_ts__",label]].copy();q=q.rename(columns={label:"label"})
  for j,c in enumerate(classes):q[f"raw_{family}_{c}"]=a[:,j]
  parts.append(q)
 return pd.concat(parts,ignore_index=True)
def _outer(root:Path,side:str,month:str,family:str,classes:tuple[str,...])->pd.DataFrame:
 x=pd.read_parquet(root/side/"oof.parquet");x=x.loc[x.oof_month.eq(month)].copy();q=x[IDENTITY+["__label_end_ts__","class_label" if family=="six" else "risk_class","realized_execution_net_ev_12h"]].copy();q=q.rename(columns={"class_label":"label","risk_class":"label"})
 for c in classes:q[f"raw_{family}_{c}"]=x[f"prob_{c}"].to_numpy()
 return q
def _calibrate(inner:pd.DataFrame,outer:pd.DataFrame,family:str,classes:tuple[str,...])->tuple[pd.DataFrame,pd.DataFrame,float]:
 cols=[f"raw_{family}_{c}" for c in classes];y=pd.Categorical(inner.label,categories=classes).codes;t=_temperature(inner[cols].to_numpy(float),y)
 for z in (inner,outer):
  p=_softmax(np.log(np.clip(z[cols].to_numpy(float),1e-8,1.)),t)
  for j,c in enumerate(classes):z[f"cal_{family}_{c}"]=p[:,j]
 return inner,outer,t
def _map(inner:pd.DataFrame,outer:pd.DataFrame,cols:list[str],name:str)->None:
 # A deliberately low-capacity diagnostic context map.  It consumes only
 # prior inner-OOF probabilities and is not the final execution-EV head.
 model=Ridge(alpha=10.0).fit(inner[cols],inner.realized_execution_net_ev_12h)
 outer[name]=model.predict(outer[cols]);inner[name]=model.predict(inner[cols])
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--labels-root",type=Path,required=True);p.add_argument("--context-root",type=Path,required=True);p.add_argument("--population",type=Path,required=True);p.add_argument("--six-root",type=Path,required=True);p.add_argument("--risk-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--only-side",choices=("long","short"));p.add_argument("--only-month",choices=("2025-03","2025-04"));a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
 streams=[];temps=[]
 for month,start in (("2025-03",pd.Timestamp("2025-03-01",tz="UTC")),("2025-04",pd.Timestamp("2025-04-01",tz="UTC"))):
  if a.only_month and month!=a.only_month:continue
  for side in ("long","short"):
   if a.only_side and side!=a.only_side:continue
   ip=a.output_root/f"inner_{side}_{month}.parquet";op=a.output_root/f"outer_{side}_{month}.parquet"
   if ip.exists() and op.exists():
    inner=pd.read_parquet(ip);outer=pd.read_parquet(op)
    t6=_temperature(inner[[f"raw_six_{c}" for c in CLASS_ORDER]].to_numpy(float),pd.Categorical(inner.six_label,categories=CLASS_ORDER).codes)
    tr=_temperature(inner[[f"raw_risk_{c}" for c in RISK_ORDER]].to_numpy(float),pd.Categorical(inner.risk_label,categories=RISK_ORDER).codes)
    streams.append(outer);temps.append({"side":side,"month":month,"six_temperature":t6,"risk_temperature":tr,"inner_oof_rows":len(inner),"reused_atomic_checkpoint":True});continue
   print(f"causal-calibration start side={side} month={month}",flush=True)
   six=six_frame(a,side);six["class_label"]=_geometry_labels(six,float(json.loads((a.six_root/side/"geometry.json").read_text())["winner"]["peak_mfe_r_threshold"]))
   risk=risk_frame(a,side);hi=_predict_inner(six,side,"six",_load_hpo(a.six_root,side),start);ri=_predict_inner(risk,side,"risk",_load_hpo(a.risk_root,side),start);inner=hi.merge(ri,on=IDENTITY+["__label_end_ts__"],validate="one_to_one",suffixes=("_six","_risk"));inner=inner.rename(columns={"label_six":"six_label","label_risk":"risk_label"});econ=pd.read_parquet(a.population,columns=["candidate_id","execution_net_ev_12h"]);inner=inner.merge(econ,on="candidate_id",validate="one_to_one").rename(columns={"execution_net_ev_12h":"realized_execution_net_ev_12h"})
   ho=_outer(a.six_root,side,month,"six",CLASS_ORDER);ro=_outer(a.risk_root,side,month,"risk",RISK_ORDER);outer=ho.merge(ro,on=IDENTITY+["__label_end_ts__","realized_execution_net_ev_12h"],validate="one_to_one",suffixes=("_six","_risk"));outer=outer.rename(columns={"label_six":"six_label","label_risk":"risk_label"})
   hi2=inner.rename(columns={"six_label":"label"}).copy();ho2=outer.rename(columns={"six_label":"label"}).copy();hi2,ho2,t6=_calibrate(hi2,ho2,"six",CLASS_ORDER);ri2=inner.rename(columns={"risk_label":"label"}).copy();ro2=outer.rename(columns={"risk_label":"label"}).copy();ri2,ro2,tr=_calibrate(ri2,ro2,"risk",RISK_ORDER)
   for c in CLASS_ORDER:outer[f"cal_six_{c}"]=ho2[f"cal_six_{c}"].to_numpy();inner[f"cal_six_{c}"]=hi2[f"cal_six_{c}"].to_numpy()
   for c in RISK_ORDER:outer[f"cal_risk_{c}"]=ro2[f"cal_risk_{c}"].to_numpy();inner[f"cal_risk_{c}"]=ri2[f"cal_risk_{c}"].to_numpy()
   raw_six=[f"raw_six_{c}" for c in CLASS_ORDER];cal_six=[f"cal_six_{c}" for c in CLASS_ORDER];raw_risk=[f"raw_risk_{c}" for c in RISK_ORDER];cal_risk=[f"cal_risk_{c}" for c in RISK_ORDER]
   for cols,name in ((raw_six,"mapped_raw_six_ev"),(cal_six,"mapped_cal_six_ev"),(raw_risk,"mapped_raw_risk_ev"),(cal_risk,"mapped_cal_risk_ev"),(raw_six+raw_risk,"mapped_raw_combined_context_ev"),(cal_six+cal_risk,"mapped_cal_combined_context_ev")):_map(inner,outer,cols,name)
   outer["raw_six_actionable"]=outer[[f"raw_six_{c}" for c in SIX_ACTION]].sum(axis=1);outer["cal_six_actionable"]=outer[[f"cal_six_{c}" for c in SIX_ACTION]].sum(axis=1);outer["raw_risk_favorable"]=outer["raw_risk_favorable_first"];outer["cal_risk_favorable"]=outer["cal_risk_favorable_first"];outer["side"]=side;outer["oof_month"]=month;_write_parquet(a.output_root/f"outer_{side}_{month}.parquet",outer);_write_parquet(a.output_root/f"inner_{side}_{month}.parquet",inner);streams.append(outer);temps.append({"side":side,"month":month,"six_temperature":t6,"risk_temperature":tr,"inner_oof_rows":len(inner)});print(f"causal-calibration done side={side} month={month}",flush=True)
 if a.only_side or a.only_month:return
 allx=pd.concat(streams,ignore_index=True);reports={}
 for family,classes in (("six",CLASS_ORDER),("risk",RISK_ORDER)):
  reports[family]={"raw":_metrics(allx.rename(columns={f"{family}_label":"label"}),classes,f"raw_{family}_"),"temperature_calibrated":_metrics(allx.rename(columns={f"{family}_label":"label"}),classes,f"cal_{family}_")}
 scores=["raw_six_actionable","cal_six_actionable","raw_risk_favorable","cal_risk_favorable","mapped_raw_six_ev","mapped_cal_six_ev","mapped_raw_risk_ev","mapped_cal_risk_ev","mapped_raw_combined_context_ev","mapped_cal_combined_context_ev"];top={s:_top(allx,s) for s in scores};latest={}
 for side in ("long","short"):
  q=allx.loc[(allx.side.eq(side))&(allx.oof_month.eq("2025-04"))];latest[side]={s:_top(q,s) for s in scores}
 r={"schema":SCHEMA,"research_status":"research_only_prior_internal_oof_non_nested_hpo","scope":"causal temperature calibration and diagnostic context mapping from prior resolved inner OOF only","promotion_caveat":"Causal versus March/April outer labels, but February model FS/HPO/geometry used labels overlapping the internal calibration era. A promotion run must nest FS/HPO/geometry inside calibration folds or use an earlier calibration era.","temperatures":temps,"classification_calibration":reports,"pooled_global_top10":top,"latest_month_side_top10":latest,"correlation_incrementality":{"raw_six_vs_risk":float(allx.raw_six_actionable.corr(allx.raw_risk_favorable)),"cal_six_vs_risk":float(allx.cal_six_actionable.corr(allx.cal_risk_favorable)),"raw_top10_jaccard":_jaccard_top(allx,"raw_six_actionable","raw_risk_favorable")},"no_final_execution_ev_head_trained":True};_write(a.output_root/"report.json",r);print(json.dumps(r,sort_keys=True,default=str))
if __name__=="__main__":main()
