#!/usr/bin/env python3
"""Frozen side-local exit/hurdle reliability ablation; research only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, shutil, sys, tempfile, time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/canonical_execution_reliability_exit_hurdle_workstream_20260730_v1.json"
OUT=ROOT/"data_perp/artifacts/canonical_execution_reliability_exit_hurdle_ablation_20260730_v1"
CHECKPOINT_VERSION="exit_hurdle_side_fold_head_v1"
SIDES=("long","short"); TOPS=(.01,.05,.10,.20); TIME="execution_decision_utc"; END="execution_label_end_utc"; NET="execution_net_ev_12h"; GROSS="execution_gross_ev_12h"; COST="execution_cost_return"; ID=("candidate_id","side_name","__symbol__","__ts__")
spec=importlib.util.spec_from_file_location("_parent_reliability",ROOT/"scripts/run_canonical_execution_reliability_ablation_v2.py")
assert spec and spec.loader
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
Geometry=parent.Geometry
class ContractError(RuntimeError): pass

def sha(p:Path)->str:return parent.sha(p)
def safe(x:Any)->Any:return parent.safe(x)
def write_json(p:Path,x:Mapping[str,Any])->None:parent.write_json(p,x)
def stable_hash(value:Any)->str:
 return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(",",":")).encode()).hexdigest()
def identity_hash(x:pd.DataFrame)->str:
 cols=[*ID,TIME,END]
 q=x.loc[:,cols].copy()
 for z in ("__ts__",TIME,END):q[z]=pd.to_datetime(q[z],utc=True).astype("int64")
 h=pd.util.hash_pandas_object(q,index=False).to_numpy(np.uint64)
 return hashlib.sha256(h.tobytes()).hexdigest()
def log_event(root:Path,event:str,**fields:Any)->None:
 root.mkdir(parents=True,exist_ok=True);row={"utc":pd.Timestamp.now(tz="UTC").isoformat(),"event":event,**safe(fields)}
 with (root/"progress.jsonl").open("a") as f:
  f.write(json.dumps(row,sort_keys=True)+"\n");f.flush();os.fsync(f.fileno())
def checkpoint_path(root:Path,architecture:str,variant:str,fold:str,side:str,head:str)->Path:
 def slug(x:str)->str:return "".join(z if z.isalnum() or z in "._-" else "_" for z in str(x))
 return root/slug(architecture)/slug(variant)/slug(fold)/slug(side)/slug(head)
def checkpointed_fit(*,root:Path,resume:bool,global_fingerprint:str,architecture:str,variant:str,fold:str,side:str,head:str,train:pd.DataFrame,valid:pd.DataFrame,features:Sequence[str],target:str,task:str,geoms:Sequence[Geometry],seed:int,stats:dict[str,int]):
 path=checkpoint_path(root,architecture,variant,fold,side,head);meta_path=path/"metadata.json";pred_path=path/"predictions.npz"
 expected={"version":CHECKPOINT_VERSION,"global_fingerprint":global_fingerprint,"architecture":architecture,"variant":variant,"fold":fold,"side":side,"head":head,"task":task,"target":target,"seed":seed,"features":list(features),"train_identity_sha256":identity_hash(train),"valid_identity_sha256":identity_hash(valid),"train_rows":len(train),"valid_rows":len(valid)}
 if meta_path.exists() or pred_path.exists():
  if not resume:raise ContractError(f"checkpoint exists and --no-resume was requested: {path}")
  if not meta_path.is_file() or not pred_path.is_file():raise ContractError(f"incomplete checkpoint: {path}")
  meta=json.loads(meta_path.read_text())
  if meta.get("expected")!=expected:raise ContractError(f"checkpoint fingerprint/identity mismatch: {path}")
  if sha(pred_path)!=meta.get("predictions_sha256"):raise ContractError(f"checkpoint payload hash mismatch: {path}")
  with np.load(pred_path,allow_pickle=False) as z:p=np.asarray(z["prediction"])
  if len(p)!=len(valid):raise ContractError(f"checkpoint prediction rows changed: {path}")
  stats["reused"]+=1;log_event(root,"REUSE",path=str(path),architecture=architecture,variant=variant,fold=fold,side=side,head=head)
  return p,meta["detail"],tuple(meta.get("classes",[])),list(meta["chosen"])
 if path.exists():raise ContractError(f"unsealed checkpoint path exists: {path}")
 path.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();log_event(root,"START",path=str(path),architecture=architecture,variant=variant,fold=fold,side=side,head=head)
 tmp_path=Path(tempfile.mkdtemp(prefix=f".{path.name}.",dir=path.parent))
 try:
  p,detail,model,chosen=fit(train,valid,features,target,task,geoms,seed);classes=tuple(np.asarray(model.classes_).astype(str)) if task in ("binary","multi") else ()
  tmp_pred=tmp_path/"predictions.npz"
  with tmp_pred.open("wb") as f:np.savez_compressed(f,prediction=np.asarray(p))
  meta={"expected":expected,"detail":safe(detail),"chosen":list(chosen),"classes":list(classes),"predictions_sha256":sha(tmp_pred),"wall_seconds":time.monotonic()-started}
  write_json(tmp_path/"metadata.json",meta);os.replace(tmp_path,path)
  stats["fitted"]+=1;log_event(root,"COMPLETE",path=str(path),wall_seconds=meta["wall_seconds"],architecture=architecture,variant=variant,fold=fold,side=side,head=head)
  return np.asarray(p),detail,classes,list(chosen)
 except Exception as e:
  shutil.rmtree(tmp_path,ignore_errors=True)
  log_event(root,"ERROR",path=str(path),error_type=type(e).__name__,error=str(e),architecture=architecture,variant=variant,fold=fold,side=side,head=head);raise
def checkpoint_fingerprint(c:Mapping[str,Any],config_path:Path,fm:Mapping[str,Any],tm:Mapping[str,Any])->tuple[str,dict[str,Any]]:
 parent_root=ROOT/c["parent_evidence"]["ablation"]
 inputs={"checkpoint_version":CHECKPOINT_VERSION,"config_sha256":sha(config_path),"feature_manifest_sha256":sha(ROOT/c["feature_input"]["artifact"]/"manifest.json"),"feature_panel_sha256":fm["outputs_sha256"]["panel.parquet"],"target_manifest_sha256":sha(ROOT/c["target_input"]["artifact"]/"manifest.json"),"target_labels_sha256":tm["outputs_sha256"]["labels.parquet"],"parent_ablation_manifest_sha256":sha(parent_root/"manifest.json"),"runner_sha256":sha(Path(__file__).resolve())}
 return stable_hash(inputs),inputs
class RunLock:
 def __init__(self,path:Path):self.path=path
 def __enter__(self):
  self.path.parent.mkdir(parents=True,exist_ok=True)
  try:
   fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
  except FileExistsError:
   payload=json.loads(self.path.read_text());pid=int(payload.get("pid",-1))
   try:os.kill(pid,0)
   except ProcessLookupError:
    pass
   except PermissionError:
    raise ContractError(f"cannot prove checkpoint lock is stale: {self.path}")
   else:raise ContractError(f"duplicate fit already running under pid {pid}")
   self.path.unlink();fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
  os.write(fd,json.dumps({"pid":os.getpid(),"started_utc":pd.Timestamp.now(tz="UTC").isoformat()}).encode());os.close(fd);return self
 def __exit__(self,*exc):
  if self.path.exists():self.path.unlink()
def sealed(root:Path,schema:str)->dict[str,Any]:
 m=root/"manifest.json";s=root/"manifest.sha256"
 if not m.is_file() or not s.is_file() or sha(m)!=s.read_text().split()[0]:raise ContractError(f"invalid seal: {root}")
 x=json.loads(m.read_text())
 if x.get("schema")!=schema:raise ContractError(f"schema mismatch: {root}")
 for n,h in x.get("outputs_sha256",{}).items():
  if not (root/n).is_file() or sha(root/n)!=h:raise ContractError(f"output hash mismatch {root/n}")
 return x
def contract(path:Path=CONFIG)->dict[str,Any]:
 c=json.loads(path.read_text())
 if c.get("schema")!="canonical_execution_reliability_exit_hurdle_workstream_v1":raise ContractError("frozen exit/hurdle v1 contract required")
 if c["unsupported_target_rule"].get("target_deployed_other_adverse_exit_attribution_only","").startswith("FORBIDDEN") is False:raise ContractError("other adverse must remain forbidden")
 if len(c["feature_contract"]["bounded_interactions"])!=5:raise ContractError("exactly five interactions required")
 return c
def outer(x:pd.DataFrame,f:Mapping[str,Any])->tuple[np.ndarray,np.ndarray]:return parent.outer_masks(x,f)
def interactions(train:pd.DataFrame,target:pd.DataFrame,c:Mapping[str,Any])->pd.DataFrame:
 sources=[z.split(" x ",1)[1] for z in c["feature_contract"]["bounded_interactions"]]
 return parent.interaction_features(train,train.base_oof_score,sources,target)
def hmetrics(y:np.ndarray,p:np.ndarray,task:str,classes:Sequence[str]|None=None)->dict[str,Any]:return parent.head_metrics(y,p,task,classes=classes)
def fit(train:pd.DataFrame,valid:pd.DataFrame,features:Sequence[str],target:str,task:str,geoms:Sequence[Geometry],seed:int):return parent.fit_head(train,valid,features,target,task,geoms,seed)
def pred(model:Any,x:pd.DataFrame,task:str)->np.ndarray:return parent._prediction(model,x,task)
def map_scores(x:pd.DataFrame):return parent.causal_map(x)

def load(c:Mapping[str,Any])->tuple[pd.DataFrame,dict[str,Any],dict[str,Any]]:
 fr=ROOT/c["feature_input"]["artifact"];tr=ROOT/c["target_input"]["artifact"]
 fm=sealed(fr,c["feature_input"]["schema"]);tm=sealed(tr,c["target_input"]["schema"])
 fields=set(c["feature_contract"]["base_fields"]); trans=[z.split(" x ",1)[1] for z in c["feature_contract"]["bounded_interactions"]]
 required=[*ID,TIME,END,NET,GROSS,COST,"candidate_month","base_oof_score","execution_exit_class","execution_exit_reason","__regime_source_execution_risk_score__",*fields,*trans]
 x=pd.read_parquet(fr/"panel.parquet",columns=list(dict.fromkeys(required)))
 y=pd.read_parquet(tr/"labels.parquet")
 for z in ("__ts__",TIME,END):x[z]=pd.to_datetime(x[z],utc=True);y[z]=pd.to_datetime(y[z],utc=True)
 ycols=[z for z in y if z.startswith("target_") or z=="label_available_at_utc"]
 x=x.merge(y.loc[:,[*ID,TIME,END,*ycols]],on=[*ID,TIME,END],how="inner",validate="one_to_one")
 if len(x)!=110730 or x.duplicated(["candidate_id","side_name"]).any():raise ContractError("feature/target identity contract")
 if not x[END].eq(x[TIME]+pd.Timedelta(hours=12)).all() or not x["label_available_at_utc"].eq(x[END]).all():raise ContractError("H12 target availability contract")
 if not np.allclose(x[GROSS]-x[COST],x[NET],atol=1e-7):raise ContractError("gross-cost-net")
 risk=x.loc[x.candidate_month.eq("2025-03"),"__regime_source_execution_risk_score__"]
 edges=np.unique(risk.quantile(np.linspace(0,1,6)).to_numpy());
 if len(edges)!=6:raise ContractError("regime quintile failure")
 edges[0],edges[-1]=-np.inf,np.inf;x["regime_execution_risk_quintile"]=pd.cut(x["__regime_source_execution_risk_score__"],edges,labels=["Q1","Q2","Q3","Q4","Q5"],include_lowest=True).astype(str)
 return x,fm,tm

PERSIST=[*ID,TIME,END,NET,GROSS,COST,"candidate_month","execution_exit_class","execution_exit_reason","regime_execution_risk_quintile","target_pre_exit_opportunity_25bps"]
def condition(x:pd.DataFrame,name:str)->np.ndarray:
 if name=="all":return np.ones(len(x),bool)
 if name=="opp":return x["_opp"].astype(bool).to_numpy()
 if name=="success":return x.target_successful_deployed_trailing.astype(bool).to_numpy()
 if name=="opp_fail":return (x["_opp"].astype(bool)&~x.target_successful_deployed_trailing.astype(bool)).to_numpy()
 if name=="no_opp":return ~x["_opp"].astype(bool).to_numpy()
 if name=="not_success":return ~x.target_successful_deployed_trailing.astype(bool).to_numpy()
 if name=="nonsevere":return ~x.target_severe_loss_100bps.astype(bool).to_numpy()
 if name=="severe":return x.target_conditional_severe_loss_mask.astype(bool).to_numpy()
 if name.startswith("class:"):return x.target_deployed_exit_economics_class.astype(str).eq(name.split(":",1)[1]).to_numpy()
 raise KeyError(name)
def targets_for(x:pd.DataFrame,architecture:str,opp_col:str|None)->list[tuple[str,str,str,str]]:
 if architecture=="H1":
  x["_opp"]=x[opp_col].astype(bool);x["_gain"]=np.maximum(x[NET],0.)
  return [("opp",opp_col,"binary","all"),("success","target_successful_deployed_trailing","binary","opp"),("gain","_gain","reg","success"),("pay_opp_failure",NET,"reg","opp_fail"),("pay_noopp",NET,"reg","no_opp")]
 if architecture=="H2":return [("class","target_deployed_exit_economics_class","multi","all"),*[("pay_"+z,NET,"reg","class:"+z) for z in cclasses()]]
 if architecture=="H3":return [("success","target_successful_deployed_trailing","binary","all"),("riskclass","target_deployed_exit_economics_class","multi","not_success"),("pay_success",NET,"reg","success"),*[("pay_"+z,NET,"reg","class:"+z) for z in cclasses()[1:]]]
 if architecture=="H4":return [("severe","target_severe_loss_100bps","binary","all"),("severity","target_conditional_severe_loss_log1p_100bps","reg","severe"),("nonsevere",NET,"reg","nonsevere")]
 raise KeyError(architecture)
def cclasses():return ("successful_trailing","trailing_nonpositive","hard_adverse","timeout")

def score_arch(dev:pd.DataFrame,apr:pd.DataFrame,c:Mapping[str,Any],architecture:str,variant:str,geoms:Sequence[Geometry],*,checkpoint_root:Path,resume:bool,global_fingerprint:str,stats:dict[str,int])->tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]],dict[str,Any]]:
 base=list(c["feature_contract"]["base_fields"]);heads=[];oof=[];recipes={};opp_col=variant if architecture=="H1" else None
 def one(train:pd.DataFrame,valid:pd.DataFrame,foldname:str,side:str,seed:int,freeze:bool=False):
  a=train.copy();b=valid.copy()
  if architecture=="H1":
   a["_opp"]=a[opp_col].astype(bool);b["_opp"]=b[opp_col].astype(bool)
   a["_gain"]=np.maximum(a[NET],0.);b["_gain"]=np.maximum(b[NET],0.)
  it=interactions(a,a,c);iv=interactions(a,b,c)
  for z in it:a[z]=it[z];b[z]=iv[z]
  features=[*base,*it.columns];out={}; classes=cclasses();local_recipes={}
  for hn,t,task,mask in targets_for(a,architecture,opp_col):
   local=a.loc[condition(a,mask)].copy()
   if len(local)<100 or (task=="binary" and local[t].nunique()<2):raise ContractError(f"unsupported {architecture}/{foldname}/{side}/{hn}")
   if task=="multi" and set(local[t].astype(str).unique())!=set(classes if hn=="class" else classes[1:]):raise ContractError(f"incomplete multiclass {hn}")
   p,d,model_classes,chosen=checkpointed_fit(root=checkpoint_root,resume=resume,global_fingerprint=global_fingerprint,architecture=architecture,variant=variant,fold=foldname,side=side,head=hn,train=local,valid=b,features=features,target=t,task=task,geoms=geoms,seed=seed,stats=stats)
   metric=b.loc[condition(b,mask)];mp=p[condition(b,mask)]
   if task=="multi":
    names=classes if hn=="class" else classes[1:];got=tuple(model_classes);
    if set(got)!=set(names):raise ContractError("class order drift")
    for name in names:out[(hn,name)]=p[:,int(np.flatnonzero(np.asarray(got)==name)[0])]
    metric_class=got
   else:out[hn]=p;metric_class=None
   heads.append({"architecture":architecture,"variant":variant,"fold":foldname,"side":side,"head":hn,"target":t,"task":task,"condition":mask,"support":len(local),**d,**hmetrics(metric[t].to_numpy(),mp,task,metric_class)})
   local_recipes[hn]={"geometry":d["geometry"],"features":list(chosen),"classes":list(model_classes),"condition":mask,"target":t,"task":task}
  raw=combine(architecture,out)
  z=b.loc[:,PERSIST].copy();z["raw_score"]=raw;z["score_available_utc"]=z[TIME];z["outer_fold"]=foldname;z["candidate_score_is_oof"]=not freeze
  return z,{"source":"pre_april_full_march_inner_hpo_pvc" if freeze else foldname,"heads":local_recipes}
 for i,f in enumerate(c["outer_folds"]):
  tr,va=outer(dev,f)
  for side in SIDES:
   z,r=one(dev.loc[tr&dev.side_name.eq(side)],dev.loc[va&dev.side_name.eq(side)],f["name"],side,20260730+i);oof.append(z);recipes[f"{f['name']}:{side}"]=r
 recipes["outer_folds"]="side-local legal decision+label-end purge; fold-local inner HPO/PVC"
 frozen=[]
 for side in SIDES:
  train=dev.loc[dev.side_name.eq(side)&dev[END].lt(pd.Timestamp("2025-04-01T00:00:00Z"))];valid=apr.loc[apr.side_name.eq(side)]
  z,r=one(train,valid,"april_frozen",side,20269999,True);frozen.append(z);recipes[f"april_frozen:{side}"]=r
 return pd.concat(oof,ignore_index=True),pd.concat(frozen,ignore_index=True),heads,recipes
def combine(a:str,h:Mapping[Any,np.ndarray])->np.ndarray:
 if a=="H1":return h["opp"]*(h["success"]*np.maximum(h["gain"],0)+(1-h["success"])*h["pay_opp_failure"])+(1-h["opp"])*h["pay_noopp"]
 if a=="H2":return sum(h[("class",z)]*h["pay_"+z] for z in cclasses())
 if a=="H3":return h["success"]*h["pay_success"]+(1-h["success"])*sum(h[("riskclass",z)]*h["pay_"+z] for z in cclasses()[1:])
 if a=="H4":return (1-h["severe"])*h["nonsevere"]-h["severe"]*(.01*np.expm1(h["severity"]))
 raise KeyError(a)

def weights(x:pd.DataFrame,col:str,f:float):return parent.global_book_weights(x,col,f)
def econ(frame:pd.DataFrame,name:str,stage:str)->list[dict[str,Any]]:
 rows=[];windows=[("aggregate",frame)]
 if "outer_fold" in frame:windows.extend((str(k),v) for k,v in frame.groupby("outer_fold"))
 if stage.startswith("april"):
  w=frame.loc[frame[TIME].ge(pd.Timestamp("2025-04-24T00:00:00Z"))];windows.append(("latest_7_decision_days",w))
 for win,x in windows:
  for kind,col in (("raw","raw_score"),("mapped","mapped_score")):
   z=x if kind=="raw" else x.loc[x.mapped_eligible]
   if not len(z):continue
   for f in TOPS:rows.append({"config":name,"stage":stage,"window":win,"score_kind":kind,"top_fraction":f,"candidate_rows":len(z),**parent.random_tie_expected(z,col,f)})
 return rows
def attribution(frame:pd.DataFrame,name:str,stage:str)->list[dict[str,Any]]:
 rows=[]
 for win,x in [("aggregate",frame),("latest_7_decision_days",frame.loc[frame[TIME].ge(pd.Timestamp("2025-04-24T00:00:00Z"))] if stage.startswith("april") else frame.iloc[:0])]:
  z=x.loc[x.mapped_eligible]
  if not len(z):continue
  for f in TOPS:
   w,meta=weights(z,"mapped_score",f);den=meta["selected_rows"]
   for dim,col in (("side","side_name"),("asset","__symbol__"),("regime","regime_execution_risk_quintile"),("exit","execution_exit_class")):
    for val,idx in z.groupby(col).groups.items():
     q=w.loc[idx];rows.append({"config":name,"stage":stage,"window":win,"top_fraction":f,"dimension":dim,"value":str(val),"candidate_rows_group":len(idx),"expected_selected_rows":float(q.sum()),"selected_book_share":float(q.sum()/den),"net_bps_contribution":float((q*z.loc[idx,NET]).sum()/den*1e4),"gross_bps_contribution":float((q*z.loc[idx,GROSS]).sum()/den*1e4),"cost_bps_contribution":float((q*z.loc[idx,COST]).sum()/den*1e4),**meta})
 return rows
def selection(march:pd.DataFrame,c:Mapping[str,Any])->dict[str,Any]:
 mapped,_=map_scores(march);vals=[]
 for f in c["outer_folds"]:
  if f["role"].startswith("architecture"):
   x=mapped.loc[mapped.outer_fold.eq(f["name"])]
   if not len(x) or not x.mapped_eligible.all():raise ContractError("selection map coverage")
   vals.append(parent.random_tie_expected(x,"mapped_score",.1)["random_tie_expected_net_bps"])
 a=np.asarray(vals);return {"mean":float(a.mean()),"std":float(a.std()),"worst":float(a.min()),"latest_fold":float(a[-1]),"objective":float(a.mean()-.5*a.std()+.25*a.min()),"fold_top10_net_bps":json.dumps(dict(zip(["selection_1","selection_2","selection_3"],vals)))}
def controls(c:Mapping[str,Any],dev:pd.DataFrame,apr:pd.DataFrame)->list[tuple[str,pd.DataFrame,pd.DataFrame]]:
 root=ROOT/c["parent_evidence"]["ablation"];m=sealed(root,"canonical_execution_reliability_ablation_v2");z=pd.read_parquet(root/"scores.parquet")
 out=[]
 for name in c["architectures"]["H0_controls"]["scores"]:
  x=z.loc[z.config.eq(name)].copy();x["candidate_score_is_oof"]=x.candidate_score_is_oof.astype(bool)
  # Rejoin authoritative v4 outcomes and use raw score only; old mapped coordinates are discarded.
  keep=[*ID,TIME,END,"raw_score","score_available_utc","outer_fold","candidate_score_is_oof"]
  x=x.loc[:,keep];q=dev.loc[:,PERSIST].merge(x.loc[x.candidate_score_is_oof],on=list(ID)+[TIME,END],how="inner",validate="one_to_one");a=apr.loc[:,PERSIST].merge(x.loc[~x.candidate_score_is_oof],on=list(ID)+[TIME,END],how="inner",validate="one_to_one");out.append(("H0__"+name,q,a))
 return out
def gates(e:pd.DataFrame,a:pd.DataFrame,winner:str,controls_:Sequence[str])->pd.DataFrame:
 q=e[(e.config.eq(winner))&(e.score_kind.eq("mapped"))&(e.top_fraction.eq(.1))];march=q[(q.stage=="march_oof")];ap=q[q.stage.str.startswith("april")];sel=march[march.window.isin(["selection_1","selection_2","selection_3"])]
 side=a[(a.config==winner)&(a.stage=="march_oof")&(a.window=="aggregate")&(a.top_fraction==.1)&(a.dimension=="side")]
 vals={"march_aggregate_positive":float(march[march.window=="aggregate"].random_tie_expected_net_bps.iloc[0])>0,"march_latest_and_worst_positive":bool((sel.random_tie_expected_net_bps>0).all()),"april_aggregate_and_latest7_positive":bool((ap[ap.window.isin(["aggregate","latest_7_decision_days"])].random_tie_expected_net_bps>0).all()),"both_side_contributions_positive":bool((side.net_bps_contribution>0).all()),"tie_selected_share_le_5pct":bool((pd.concat([march[march.window=="aggregate"],sel]).cutoff_tie_selected_share<=.05).all()),"better_than_controls":False,"new_untouched_forward_evidence":False}
 win=float(march[march.window=="aggregate"].random_tie_expected_net_bps.iloc[0]);cs=[]
 for name in controls_:
  z=e[(e.config==name)&(e.stage=="march_oof")&(e.window=="aggregate")&(e.score_kind=="mapped")&(e.top_fraction==.1)]
  if len(z):cs.append(float(z.iloc[0].random_tie_expected_net_bps))
 vals["better_than_controls"]=bool(cs and win>max(cs));return pd.DataFrame([{"gate":k,"passed":v} for k,v in vals.items()])
def run(output:Path=OUT,config_path:Path=CONFIG,*,checkpoints:Path|None=None,resume:bool=True,_locked:bool=False)->dict[str,Any]:
 checkpoint_root=checkpoints or output.parent/f"{output.name}_checkpoints"
 if not _locked:
  with RunLock(checkpoint_root/"run.lock"):
   return run(output,config_path,checkpoints=checkpoint_root,resume=resume,_locked=True)
 if output.exists():raise FileExistsError(output)
 c=contract(config_path);x,fm,tm=load(c);dev=x[x.candidate_month.eq("2025-03")].copy();apr=x[x.candidate_month.eq("2025-04")].copy()
 fingerprint,fingerprint_inputs=checkpoint_fingerprint(c,config_path,fm,tm);checkpoint_root.mkdir(parents=True,exist_ok=True);contract_path=checkpoint_root/"checkpoint_contract.json"
 declared={"schema":CHECKPOINT_VERSION,"fingerprint":fingerprint,"fingerprint_inputs":fingerprint_inputs}
 if contract_path.exists():
  if json.loads(contract_path.read_text())!=declared:raise ContractError(f"checkpoint root fingerprint changed: {checkpoint_root}")
 elif any(checkpoint_root.rglob("metadata.json")):
  raise ContractError("checkpoint payload exists without an immutable root contract")
 else:write_json(contract_path,declared)
 stats={"fitted":0,"reused":0};log_event(checkpoint_root,"RUN_START",fingerprint=fingerprint,resume=resume,pid=os.getpid())
 if len(dev)!=41472 or len(apr)!=69258:raise ContractError("population drift")
 geoms=tuple(Geometry(**g) for g in c["catboost_hpo"]["geometries"]);items=[];heads=[];sel=[]
 for name,q,a in controls(c,dev,apr):items.append((name,q,a,[],{"rule":"sealed H0 raw scores reused; old mapped coordinates discarded and remapped"}));sel.append({"config":name,"architecture":"H0","variant":"sealed_control"})
 for v in c["architectures"]["H1_cost_aware_opportunity"]["opportunity_variants"]:
  q,a,h,r=score_arch(dev,apr,c,"H1",v,geoms,checkpoint_root=checkpoint_root,resume=resume,global_fingerprint=fingerprint,stats=stats);s=selection(q,c);name="H1__"+v;items.append((name,q,a,h,r));heads.extend(h);sel.append({"config":name,"architecture":"H1","variant":v,**s})
 primary="H1__"+c["architectures"]["H1_cost_aware_opportunity"]["primary"]
 for hname,arch in (("H2","H2"),("H3","H3"),("H4","H4")):
  q,a,h,r=score_arch(dev,apr,c,arch,"primary25",geoms,checkpoint_root=checkpoint_root,resume=resume,global_fingerprint=fingerprint,stats=stats);s=selection(q,c);items.append((hname,q,a,h,r));heads.extend(h);sel.append({"config":hname,"architecture":arch,"variant":"primary25",**s})
 learned=[z for z in sel if z.get("objective") is not None];winner=sorted(learned,key=lambda z:(-z["objective"],z["config"]))[0]["config"]
 scores=[];audits=[];eco=[];attr=[];recipes={}
 for name,q,a,h,r in items:
  both=pd.concat([q,a],ignore_index=True);m,au=map_scores(both);m["config"]=name;au["config"]=name;scores.append(m);audits.append(au);eco.extend(econ(m[m.candidate_month.eq("2025-03")],name,"march_oof"));eco.extend(econ(m[m.candidate_month.eq("2025-04")],name,"april_frozen_diagnostic"));attr.extend(attribution(m[m.candidate_month.eq("2025-03")],name,"march_oof"));attr.extend(attribution(m[m.candidate_month.eq("2025-04")],name,"april_frozen_diagnostic"));recipes[name]=r
 E=pd.DataFrame(eco);A=pd.DataFrame(attr);G=gates(E,A,winner,[z[0] for z in items if z[0].startswith("H0__")])
 stage=Path(tempfile.mkdtemp(prefix="."+output.name+".",dir=output.parent))
 try:
  outputs={"scores.parquet":pd.concat(scores,ignore_index=True),"head_metrics.csv":pd.DataFrame(heads),"selection.csv":pd.DataFrame(sel),"mapping_audit.csv":pd.concat(audits,ignore_index=True),"economics.csv":E,"global_book_attribution.csv":A,"promotion_gates.csv":G,"freeze_recipes.json":safe(recipes)}
  for n,v in outputs.items():
   if n.endswith(".parquet"):v.to_parquet(stage/n,index=False,compression="zstd")
   elif n.endswith(".json"):write_json(stage/n,v)
   else:v.to_csv(stage/n,index=False)
  report={"schema":"canonical_execution_reliability_exit_hurdle_ablation_v1","status":"RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY","promotion_eligible":False,"winner_by_march_selection":winner,"feature_input":{"path":str(ROOT/c["feature_input"]["artifact"]),"manifest_sha256":sha(ROOT/c["feature_input"]["artifact"]/"manifest.json"),"panel_sha256":fm["outputs_sha256"]["panel.parquet"]},"target_input":{"path":str(ROOT/c["target_input"]["artifact"]),"manifest_sha256":sha(ROOT/c["target_input"]["artifact"]/"manifest.json"),"labels_sha256":tm["outputs_sha256"]["labels.parquet"]},"config":{"path":str(config_path.resolve()),"sha256":sha(config_path)},"checkpointing":{"root":str(checkpoint_root.resolve()),"fingerprint":fingerprint,"contract_sha256":sha(contract_path),"fitted_this_run":stats["fitted"],"reused_this_run":stats["reused"],"sealed_checkpoint_count":sum(1 for _ in checkpoint_root.rglob("metadata.json"))},"outputs_sha256":{n:sha(stage/n) for n in outputs},"runner":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())},"limitations":["Other adverse exit is attribution only, never a standalone head.","April is reused frozen diagnostic evidence, not promotion evidence."]}
  write_json(stage/"manifest.json",report);(stage/"manifest.sha256").write_text(sha(stage/"manifest.json")+"  manifest.json\n");os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 log_event(checkpoint_root,"RUN_COMPLETE",winner=winner,fitted=stats["fitted"],reused=stats["reused"],artifact=str(output));return report
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--checkpoints",type=Path,default=None);p.add_argument("--no-resume",action="store_true");a=p.parse_args(argv)
 root=a.checkpoints or a.output_dir.parent/f"{a.output_dir.name}_checkpoints"
 try:result=run(a.output_dir,a.config,checkpoints=root,resume=not a.no_resume)
 except Exception as e:log_event(root,"RUN_ERROR",error_type=type(e).__name__,error=str(e));raise
 print(json.dumps(safe(result),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
