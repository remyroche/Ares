#!/usr/bin/env python3
"""Immutable research-only bounded adverse-risk overlay ablation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/bounded_adverse_risk_overlay_workstream_20260730_v1.json"
OUT=ROOT/"data_perp/artifacts/bounded_adverse_risk_overlay_ablation_20260730_v1"
SOURCE=ROOT/"data_perp/artifacts/canonical_execution_reliability_exit_hurdle_ablation_20260730_v1"
CKPT=ROOT/"data_perp/artifacts/canonical_execution_reliability_exit_hurdle_ablation_20260730_v1_checkpoints"
spec=importlib.util.spec_from_file_location("_sealed_exit_hurdle",ROOT/"scripts/run_canonical_execution_reliability_exit_hurdle_ablation.py")
assert spec and spec.loader
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
ID=list(H.ID); TIME=H.TIME; END=H.END; NET=H.NET; GROSS=H.GROSS; COST=H.COST; SIDES=H.SIDES; TOPS=H.TOPS
class ContractError(RuntimeError):pass
def sha(p:Path)->str:return H.sha(p)
def write_json(p:Path,x:Any)->None:H.write_json(p,x)
def safe(x:Any)->Any:return H.safe(x)
def lockfile(p:Path):
 class L:
  def __enter__(s):
   p.parent.mkdir(parents=True,exist_ok=True)
   try:s.fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
   except FileExistsError:raise ContractError(f"duplicate overlay run lock: {p}")
   os.write(s.fd,json.dumps({"pid":os.getpid()}).encode());os.close(s.fd);return s
  def __exit__(s,*a):
   if p.exists():p.unlink()
 return L()
def sealed(path:Path,schema:str|None=None)->dict[str,Any]:
 m=path/"manifest.json"; seal=path/"manifest.sha256"
 if not m.is_file() or not seal.is_file() or sha(m)!=seal.read_text().split()[0]:raise ContractError(f"invalid sealed artifact: {path}")
 x=json.loads(m.read_text())
 if schema and x.get("schema")!=schema:raise ContractError(f"schema mismatch: {path}")
 for name,digest in x.get("outputs_sha256",{}).items():
  if not (path/name).is_file() or sha(path/name)!=digest:raise ContractError(f"sealed output hash mismatch: {path/name}")
 return x
def contract(path:Path=CONFIG)->dict[str,Any]:
 c=json.loads(path.read_text())
 if c.get("schema")!="bounded_adverse_risk_overlay_workstream_v1":raise ContractError("overlay v1 config required")
 if c["lambda_grid"]!=[.25,.5,1.] or c["components"]!=["h2_hard_adverse","h4_severe"]:raise ContractError("fixed compact overlay grid required")
 if any(k not in {"opportunity","capture","timing","wait"} for k in c["forbidden"]):raise ContractError("unexpected feature/action exclusion")
 return c
def source_contract(c:Mapping[str,Any])->tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
 if ROOT/c["sealed_exit_hurdle_artifact"]!=SOURCE or ROOT/c["sealed_checkpoint_root"]!=CKPT:raise ContractError("exact sealed source paths required")
 sm=sealed(SOURCE,"canonical_execution_reliability_exit_hurdle_ablation_v1"); parent=sealed(ROOT/c["residual_control"]["artifact"],"canonical_execution_reliability_ablation_v2")
 if sm["runner"]["sha256"]!=sha(ROOT/"scripts/run_canonical_execution_reliability_exit_hurdle_ablation.py"):raise ContractError("source runner hash drift")
 cc=H.contract();x,fm,tm=H.load(cc);fingerprint,inputs=H.checkpoint_fingerprint(cc,ROOT/"configs/canonical_execution_reliability_exit_hurdle_workstream_20260730_v1.json",fm,tm)
 root_contract=json.loads((CKPT/"checkpoint_contract.json").read_text())
 if root_contract!={"schema":H.CHECKPOINT_VERSION,"fingerprint":fingerprint,"fingerprint_inputs":inputs}:raise ContractError("checkpoint root identity drift")
 if sm["checkpointing"]["fingerprint"]!=fingerprint or sm["checkpointing"]["contract_sha256"]!=sha(CKPT/"checkpoint_contract.json"):raise ContractError("source checkpoint binding drift")
 return cc,sm,parent
def valid_for(x:pd.DataFrame,cc:Mapping[str,Any],fold:str,side:str)->tuple[pd.DataFrame,pd.DataFrame]:
 if fold=="april_frozen":
  train=x.loc[x.candidate_month.eq("2025-03")&x.side_name.eq(side)&x[END].lt(pd.Timestamp("2025-04-01T00:00:00Z"))]
  valid=x.loc[x.candidate_month.eq("2025-04")&x.side_name.eq(side)]
 else:
  dev=x.loc[x.candidate_month.eq("2025-03")]; f=next(q for q in cc["outer_folds"] if q["name"]==fold);tr,va=H.outer(dev,f)
  train=dev.loc[tr&dev.side_name.eq(side)];valid=dev.loc[va&dev.side_name.eq(side)]
 return train.copy(),valid.copy()
def payload(arch:str,fold:str,side:str,head:str,train:pd.DataFrame,valid:pd.DataFrame)->tuple[np.ndarray,dict[str,Any]]:
 p=CKPT/arch/"primary25"/fold/side/head; mp=p/"metadata.json"; pp=p/"predictions.npz"
 if not mp.is_file() or not pp.is_file():raise ContractError(f"missing checkpoint: {p}")
 m=json.loads(mp.read_text());e=m.get("expected",{})
 checks={"architecture":arch,"variant":"primary25","fold":fold,"side":side,"head":head,"global_fingerprint":json.loads((CKPT/"checkpoint_contract.json").read_text())["fingerprint"],"train_identity_sha256":H.identity_hash(train),"valid_identity_sha256":H.identity_hash(valid),"train_rows":len(train),"valid_rows":len(valid)}
 if any(e.get(k)!=v for k,v in checks.items()):raise ContractError(f"checkpoint identity mismatch: {p}")
 if sha(pp)!=m.get("predictions_sha256"):raise ContractError(f"checkpoint payload hash mismatch: {p}")
 with np.load(pp,allow_pickle=False) as z:a=np.asarray(z["prediction"])
 if len(a)!=len(valid):raise ContractError(f"checkpoint OOS length mismatch: {p}")
 return a,m
def bounds(train:pd.DataFrame)->dict[str,float]:
 lo,hi=contract()["clip_quantiles"]
 hard=train.loc[train.target_deployed_exit_economics_class.astype(str).eq("hard_adverse"),NET].astype(float)
 severe=-.01*np.expm1(train.loc[train.target_conditional_severe_loss_mask.astype(bool),"target_conditional_severe_loss_log1p_100bps"].astype(float))
 if len(hard)<100 or len(severe)<100:raise ContractError("insufficient train-only clip support")
 return {"h2_lower":float(np.quantile(hard,lo)),"h2_upper":float(min(0.,np.quantile(hard,hi))),"h4_lower":float(np.quantile(severe,lo)),"h4_upper":float(min(0.,np.quantile(severe,hi))),"hard_train_support":len(hard),"severe_train_support":len(severe)}
def reconstructed(x:pd.DataFrame,cc:Mapping[str,Any])->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
 rows=[];metrics=[];clips=[]
 for fold in [q["name"] for q in cc["outer_folds"]]+["april_frozen"]:
  for side in SIDES:
   train,valid=valid_for(x,cc,fold,side); cp=bounds(train)
   hard_train=train.loc[train.target_deployed_exit_economics_class.astype(str).eq("hard_adverse")].copy(); severe_train=train.loc[train.target_conditional_severe_loss_mask.astype(bool)].copy()
   cl,cm=payload("H2",fold,side,"class",train,valid);pay,pm=payload("H2",fold,side,"pay_hard_adverse",hard_train,valid);sev,sm=payload("H4",fold,side,"severe",train,valid);mag,mm=payload("H4",fold,side,"severity",severe_train,valid)
   classes=np.asarray(cm["classes"],str); hit=np.flatnonzero(classes=="hard_adverse")
   if cl.ndim!=2 or len(hit)!=1 or tuple(classes)!=tuple(sorted(H.cclasses())):raise ContractError("H2 class order drift")
   hp=cl[:,int(hit[0])]; hraw=hp*np.minimum(pay,0.); hclip=np.clip(hraw,cp["h2_lower"],cp["h2_upper"])
   sraw=-sev*.01*np.expm1(np.maximum(mag,0.)); sclip=np.clip(sraw,cp["h4_lower"],cp["h4_upper"])
   z=valid.loc[:,H.PERSIST].copy();z["outer_fold"]=fold;z["candidate_score_is_oof"]=fold!="april_frozen";z["score_available_utc"]=z[TIME];z["h2_hard_adverse_probability"]=hp;z["h2_pay_hard_adverse"]=pay;z["h2_risk_raw"]=hraw;z["h2_risk_contribution"]=hclip;z["h4_severe_probability"]=sev;z["h4_conditional_severity_log1p"]=mag;z["h4_risk_raw"]=sraw;z["h4_risk_contribution"]=sclip
   rows.append(z); clips.append({"fold":fold,"side":side,**cp})
   for name,y,p,task,cond,meta in [("h2_hard_adverse_probability",valid.target_deployed_exit_economics_class.astype(str).eq("hard_adverse"),hp,"binary",np.ones(len(valid),bool),cm),("h2_pay_hard_adverse",valid[NET],pay,"reg",valid.target_deployed_exit_economics_class.astype(str).eq("hard_adverse").to_numpy(),pm),("h4_severe_probability",valid.target_severe_loss_100bps,sev,"binary",np.ones(len(valid),bool),sm),("h4_conditional_severity",valid.target_conditional_severe_loss_log1p_100bps,mag,"reg",valid.target_conditional_severe_loss_mask.astype(bool).to_numpy(),mm)]:
    yy=np.asarray(y)[cond];pp=np.asarray(p)[cond];metrics.append({"fold":fold,"side":side,"head":name,"support":len(yy),"checkpoint_sha256":sha(CKPT/("H2" if name.startswith("h2") else "H4")/"primary25"/fold/side/("class" if name=="h2_hard_adverse_probability" else "pay_hard_adverse" if name=="h2_pay_hard_adverse" else "severe" if name=="h4_severe_probability" else "severity")/"predictions.npz"),**H.hmetrics(yy,pp,task)})
 return pd.concat(rows,ignore_index=True),pd.DataFrame(metrics),pd.DataFrame(clips)
def control(x:pd.DataFrame,c:Mapping[str,Any])->pd.DataFrame:
 p=ROOT/c["residual_control"]["artifact"]/"scores.parquet";z=pd.read_parquet(p);z=z.loc[z.config.eq(c["residual_control"]["config"])].copy();z.candidate_score_is_oof=z.candidate_score_is_oof.astype(bool)
 keep=[*ID,TIME,END,"raw_score","outer_fold","candidate_score_is_oof","score_available_utc"]
 q=x.loc[:,H.PERSIST].merge(z.loc[:,keep],on=[*ID,TIME,END],how="inner",validate="one_to_one")
 if len(q)!=len(z):raise ContractError("residual control OOS join coverage mismatch")
 q.rename(columns={"raw_score":"residual_control_raw"},inplace=True);return q
def variants(base:pd.DataFrame,c:Mapping[str,Any])->list[tuple[str,pd.DataFrame]]:
 out=[]
 for name,kind,lam in [("residual_control","control",0.)]+[(f"{k}_lambda_{l:g}",k,l) for k in ("h2","h4","combined") for l in c["lambda_grid"]]:
  z=base.copy(); r=np.zeros(len(z)) if kind=="control" else z.h2_risk_contribution.to_numpy() if kind=="h2" else z.h4_risk_contribution.to_numpy() if kind=="h4" else z.h2_risk_contribution.to_numpy()+z.h4_risk_contribution.to_numpy()
  z["overlay_kind"]=kind;z["lambda"]=lam;z["overlay_risk_contribution"]=r;z["raw_score"]=z.residual_control_raw+lam*r;out.append((name,z))
 return out
def selection(x:pd.DataFrame,c:Mapping[str,Any])->dict[str,Any]:
 vals=[]
 for f in c["selection"]["folds"]:
  q=x.loc[x.outer_fold.eq(f)&x.mapped_eligible]
  if not len(q):raise ContractError("selection mapping coverage")
  vals.append(H.parent.random_tie_expected(q,"mapped_score",.1)["random_tie_expected_net_bps"])
 a=np.asarray(vals);return {"mean":float(a.mean()),"std":float(a.std()),"worst":float(a.min()),"latest_fold":float(a[-1]),"objective":float(a.mean()-.5*a.std()+.25*a.min()),"fold_top10_net_bps":json.dumps(dict(zip(c["selection"]["folds"],vals)))}
def econ(x:pd.DataFrame,name:str,stage:str)->list[dict[str,Any]]:return H.econ(x,name,stage)
def attr(x:pd.DataFrame,name:str,stage:str)->list[dict[str,Any]]:return H.attribution(x,name,stage)
def transport(mapped:pd.DataFrame,name:str)->list[dict[str,Any]]:
 out=[]
 for stage,q in (("march_oof",mapped.loc[mapped.candidate_month.eq("2025-03")]),("april_frozen_diagnostic",mapped.loc[mapped.candidate_month.eq("2025-04")])):
  for win,w in [("aggregate",q),*(list(q.groupby("outer_fold")) if stage=="march_oof" else []),("latest_7_decision_days",q.loc[q[TIME].ge(pd.Timestamp("2025-04-24T00:00:00Z"))] if stage.startswith("april") else q.iloc[:0])]:
   if not len(w):continue
   for side,s in [("pooled",w),*list(w.groupby("side_name"))]:
    out.append({"config":name,"stage":stage,"window":str(win),"side":str(side),"rows":len(s),"h2_mean_bps":float(s.h2_risk_contribution.mean()*1e4),"h4_mean_bps":float(s.h4_risk_contribution.mean()*1e4),"overlay_mean_bps":float(s.overlay_risk_contribution.mean()*1e4),"residual_overlay_pearson":float(s.residual_control_raw.corr(s.overlay_risk_contribution)) if s.overlay_risk_contribution.nunique()>1 else np.nan,"residual_overlay_spearman":float(s.residual_control_raw.corr(s.overlay_risk_contribution,method="spearman")) if s.overlay_risk_contribution.nunique()>1 else np.nan,"raw_score_changed_rate":float((s.raw_score!=s.residual_control_raw).mean())})
 return out
def turnover(mapped:pd.DataFrame,name:str)->list[dict[str,Any]]:
 if name=="residual_control":return []
 out=[]
 arm_all=mapped.loc[mapped.config.eq(name)]
 for stage,q in (("march_oof",mapped.loc[mapped.candidate_month.eq("2025-03")]),("april_frozen_diagnostic",mapped.loc[mapped.candidate_month.eq("2025-04")])):
  for win,w in [("aggregate",q),*(list(q.groupby("outer_fold")) if stage=="march_oof" else []),("latest_7_decision_days",q.loc[q[TIME].ge(pd.Timestamp("2025-04-24T00:00:00Z"))] if stage.startswith("april") else q.iloc[:0])]:
   if not len(w):continue
   arm=arm_all.loc[(arm_all.candidate_month.eq("2025-03") if stage=="march_oof" else arm_all.candidate_month.eq("2025-04"))]
   if str(win)!="aggregate": arm=arm.loc[arm.outer_fold.eq(win)] if stage=="march_oof" else arm.loc[arm[TIME].ge(pd.Timestamp("2025-04-24T00:00:00Z"))]
   arm=arm.loc[arm.mapped_eligible];ctl=mapped.loc[(mapped.config=="residual_control")&(mapped.candidate_month.eq("2025-03") if stage=="march_oof" else mapped.candidate_month.eq("2025-04"))]
   if str(win)!="aggregate": ctl=ctl.loc[ctl.outer_fold.eq(win)] if stage=="march_oof" else ctl.loc[ctl[TIME].ge(pd.Timestamp("2025-04-24T00:00:00Z"))]
   ctl=ctl.loc[ctl.mapped_eligible]
   if len(arm)!=len(ctl):continue
   wa,ma=H.weights(arm,"mapped_score",.1);wc,mc=H.weights(ctl,"mapped_score",.1); key=[*ID,TIME,END]; aa=arm.loc[:,key].copy();aa["w"]=wa.to_numpy();bb=ctl.loc[:,key].copy();bb["w"]=wc.to_numpy();j=aa.merge(bb,on=key,how="inner",validate="one_to_one",suffixes=("_arm","_control"));overlap=float(np.minimum(j.w_arm,j.w_control).sum()/ma["selected_rows"]);out.append({"config":name,"stage":stage,"window":str(win),"top_fraction":.1,"expected_overlap":overlap,"turnover":1-overlap})
 return out
def gates(E:pd.DataFrame,A:pd.DataFrame,winner:str)->pd.DataFrame:
 w=E.loc[(E.config==winner)&(E.score_kind=="mapped")&(E.top_fraction==.1)];m=w.loc[w.stage=="march_oof"];ap=w.loc[w.stage.str.startswith("april")];sel=m.loc[m.window.isin(["selection_1","selection_2","selection_3"])]; side=A.loc[(A.config==winner)&(A.stage=="march_oof")&(A.window=="aggregate")&(A.top_fraction==.1)&(A.dimension=="side")];control=float(E.loc[(E.config=="residual_control")&(E.stage=="march_oof")&(E.window=="aggregate")&(E.score_kind=="mapped")&(E.top_fraction==.1),"random_tie_expected_net_bps"].iloc[0]);win=float(m.loc[m.window=="aggregate","random_tie_expected_net_bps"].iloc[0])
 d={"march_aggregate_positive":win>0,"march_latest_and_worst_positive":bool((sel.random_tie_expected_net_bps>0).all()),"april_aggregate_and_latest7_positive":bool((ap.loc[ap.window.isin(["aggregate","latest_7_decision_days"])].random_tie_expected_net_bps>0).all()),"both_side_contributions_positive":bool(len(side)==2 and (side.net_bps_contribution>0).all()),"tie_selected_share_le_5pct":bool((pd.concat([m.loc[m.window=="aggregate"],sel]).cutoff_tie_selected_share<=.05).all()),"better_than_residual_control":bool(winner!="residual_control" and win>control),"untouched_forward_absent_no_promotion_or_replay":True}
 return pd.DataFrame([{"gate":k,"passed":v} for k,v in d.items()])
def run(output:Path=OUT,config_path:Path=CONFIG)->dict[str,Any]:
 if output.exists():raise FileExistsError(output)
 c=contract(config_path);cc,sm,pm=source_contract(c);x,_,_=H.load(cc);comp,heads,clips=reconstructed(x,cc);b=comp.merge(control(x,c),on=[*ID,TIME,END,H.NET,H.GROSS,H.COST,"candidate_month","execution_exit_class","execution_exit_reason","regime_execution_risk_quintile","target_pre_exit_opportunity_25bps","outer_fold","candidate_score_is_oof","score_available_utc"],how="inner",validate="one_to_one")
 if len(b)!=len(comp):raise ContractError("component/control strict OOS join mismatch")
 allm=[];eco=[];attrs=[];audits=[];selection_rows=[];trans=[];turn=[]
 for name,z in variants(b,c):
  mapped,audit=H.parent.causal_map(z,pooled_min=c["mapping"]["pooled_min"],side_min=c["mapping"]["side_min"]);mapped["config"]=name;audit["config"]=name;allm.append(mapped);audits.append(audit);selection_rows.append({"config":name,**selection(mapped.loc[mapped.candidate_month.eq("2025-03")],c)});eco+=econ(mapped.loc[mapped.candidate_month.eq("2025-03")],name,"march_oof") + econ(mapped.loc[mapped.candidate_month.eq("2025-04")],name,"april_frozen_diagnostic");attrs+=attr(mapped.loc[mapped.candidate_month.eq("2025-03")],name,"march_oof") + attr(mapped.loc[mapped.candidate_month.eq("2025-04")],name,"april_frozen_diagnostic");trans+=transport(mapped,name)
 M=pd.concat(allm,ignore_index=True); winner=sorted(selection_rows,key=lambda r:(-r["objective"],r["config"]))[0]["config"]
 for name in sorted(M.config.unique()):turn+=turnover(M,name)
 E=pd.DataFrame(eco);A=pd.DataFrame(attrs); rec=[]
 for _,r in E.iterrows():rec.append({"config":r.config,"stage":r.stage,"window":r.window,"score_kind":r.score_kind,"top_fraction":r.top_fraction,"gross_minus_cost_minus_net_bps":float(r.random_tie_expected_gross_bps-r.random_tie_expected_cost_bps-r.random_tie_expected_net_bps)})
 stage=Path(tempfile.mkdtemp(prefix="."+output.name+".",dir=output.parent))
 try:
  outputs={"scores.parquet":M,"head_metrics.csv":heads,"clip_bounds.csv":clips,"selection.csv":pd.DataFrame(selection_rows),"mapping_audit.csv":pd.concat(audits,ignore_index=True),"economics.csv":E,"global_book_attribution.csv":A,"overlay_transport.csv":pd.DataFrame(trans),"overlay_turnover.csv":pd.DataFrame(turn),"reconciliation.csv":pd.DataFrame(rec),"promotion_gates.csv":gates(E,A,winner)}
  for n,v in outputs.items():v.to_parquet(stage/n,index=False,compression="zstd") if n.endswith(".parquet") else v.to_csv(stage/n,index=False)
  manifest={"schema":"bounded_adverse_risk_overlay_ablation_v1","status":"RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY","promotion_eligible":False,"winner_by_march_selection":winner,"config":{"path":str(config_path.resolve()),"sha256":sha(config_path)},"parents":{"sealed_exit_hurdle_manifest_sha256":sha(SOURCE/"manifest.json"),"residual_control_manifest_sha256":sha(ROOT/c["residual_control"]["artifact"]/"manifest.json"),"checkpoint_contract_sha256":sha(CKPT/"checkpoint_contract.json"),"checkpoint_root":str(CKPT.resolve()),"source_runner_sha256":sha(ROOT/"scripts/run_canonical_execution_reliability_exit_hurdle_ablation.py")},"runner":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())},"checkpoint_payload_count":40,"outputs_sha256":{n:sha(stage/n) for n in outputs},"limitations":["Bounded overlay only; it does not replace the residual control EV.","All H2/H4 predictions are reused from identity-validated sealed checkpoints.","April is frozen diagnostic evidence; no untouched forward evidence exists, so no promotion or portfolio replay."]}
  write_json(stage/"manifest.json",manifest);(stage/"manifest.sha256").write_text(sha(stage/"manifest.json")+"  manifest.json\n");os.replace(stage,output)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return manifest
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--config",type=Path,default=CONFIG);a=p.parse_args(argv)
 with lockfile(a.output_dir.parent/(a.output_dir.name+".lock")):print(json.dumps(safe(run(a.output_dir,a.config)),indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
